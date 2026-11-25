"""Lock matched Validation-only GraphLocal-v1 objective ablations.

This program creates commands and a byte-bound source snapshot. It never
trains or evaluates a model. Formal Test is disabled in every command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"
SNAPSHOT = AREA / "source_snapshot"
ARCHIVE = ROOT / "results/hctf_strict_size_ood"
WARM = ROOT / "checkpoints/hctf_strict_size_ood"
DOMAINS = ("enzymes", "fb15k237", "sbm")
SEEDS = (17, 42, 73)
VARIANTS = {
    "A_both": {"caid": True, "macpsg": True},
    "B_caid_only": {"caid": True, "macpsg": False},
    "C_macpsg_only": {"caid": False, "macpsg": True},
    "D_neither": {"caid": False, "macpsg": False},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_option(command: list[str], flag: str, value: str) -> None:
    i = command.index(flag)
    command[i+1] = value


def snapshot() -> dict[str, str]:
    if SNAPSHOT.exists():
        raise RuntimeError("source snapshot already exists; do not overwrite a locked source")
    shutil.copytree(ROOT / "deepgd", SNAPSHOT / "deepgd",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, SNAPSHOT / path.name)
    shutil.copy2(AREA / "src/train_graphlocal_v1.py", SNAPSHOT / "train_graphlocal_v1.py")
    return {p.relative_to(SNAPSHOT).as_posix(): sha(p)
            for p in sorted(SNAPSHOT.rglob("*.py"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    cli = parser.parse_args()
    lock_path = AREA / "configs/VALIDATION_PROTOCOL_LOCK.json"
    if lock_path.exists():
        raise RuntimeError(f"protocol already locked: {lock_path}")
    split_lock = AREA / "manifests/SPLIT_LOCK.json"
    if not split_lock.exists():
        raise RuntimeError("metadata-only split manifests must be created first")
    source_hashes = {} if cli.inspect_only else snapshot()
    commands = []
    warm_hashes = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            warm = WARM / f"strictsize_{domain}_baseline_s{seed}_best.pt"
            if not warm.exists():
                raise FileNotFoundError(warm)
            warm_hashes[f"{domain}_s{seed}"] = sha(warm)
            source = ARCHIVE / f"strictsize_{domain}_adaptive_supervised_full_s{seed}_command.json"
            if not source.exists():
                raise FileNotFoundError(source)
            historical = json.loads(source.read_text(encoding="utf-8-sig"))
            if "--split-mode" not in historical or historical[historical.index("--split-mode")+1] != "size_ood":
                raise RuntimeError(f"historical command does not use size_ood: {source}")
            for name, switches in VARIANTS.items():
                command = list(historical)
                command[0] = sys.executable
                command[1] = str(SNAPSHOT / "train_graphlocal_v1.py")
                run_name = f"HCTF-GD-GraphLocal-v1_{domain}_s{seed}_{name}"
                set_option(command, "--run-name", run_name)
                set_option(command, "--save-dir", str(AREA / "checkpoints"))
                set_option(command, "--eval-csv", str(AREA / "metrics" / f"{run_name}_TEST_RESERVED.csv"))
                set_option(command, "--init-ckpt", str(warm))
                set_option(command, "--lambda-caid", "0.06" if switches["caid"] else "0")
                for flag, weight in (("--lambda-cpsg-proto", "0.035"),
                                     ("--lambda-cpsg-sil", "0.14"),
                                     ("--lambda-cpsg-sep", "0.18")):
                    set_option(command, flag, weight if switches["macpsg"] else "0")
                # Keep all input-feature switches enabled and all architecture
                # dimensions identical. Only objective coefficients change.
                for required in ("--use-caid", "--use-cpsg", "--use-macpsg"):
                    if required not in command:
                        raise RuntimeError(f"missing matched-feature switch: {required}")
                command += ["--graph-local-community-objectives", "--skip-final-test",
                            "--history-dir", str(AREA / "logs")]
                config = {"run_name": run_name, "domain": domain, "seed": seed,
                          "variant": name, "objectives": switches,
                          "mode": "Train/Validation only", "command": command,
                          "warm_checkpoint_sha256": warm_hashes[f"{domain}_s{seed}"],
                          "split_manifest_sha256": sha(AREA / "manifests" / f"{domain}_s{seed}_size_ood.csv")}
                commands.append(config)
    if cli.inspect_only:
        print(f"Inspected {len(commands)} commands; no source snapshot or lock written")
        return
    for config in commands:
        path = AREA / "configs" / f"{config['run_name']}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    lock = {"model_id": "HCTF-GD-GraphLocal-v1",
            "status": "LOCKED_VALIDATION_ONLY",
            "domains": DOMAINS, "seeds": SEEDS, "variants": VARIANTS,
            "historical_budget": {"backbone_epochs": 60, "adapter_epochs": 20,
                                  "adapter_train_batches_per_epoch": 24},
            "source_snapshot": str(SNAPSHOT), "source_hashes": source_hashes,
            "split_lock_sha256": sha(split_lock), "warm_checkpoint_hashes": warm_hashes,
            "command_count": len(commands),
            "test_evaluation": "disabled; --skip-final-test in every command",
            "git_commit": "UNAVAILABLE_NO_HEAD"}
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Locked {len(commands)} Validation-only runs at {lock_path}")


if __name__ == "__main__":
    main()
