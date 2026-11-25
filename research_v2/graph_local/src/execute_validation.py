"""Execute one locked Train/Validation job; never run Strict Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["enzymes", "fb15k237", "sbm"])
    parser.add_argument("--seed", required=True, type=int, choices=[17, 42, 73])
    parser.add_argument("--variant", required=True,
                        choices=["A_both", "B_caid_only", "C_macpsg_only", "D_neither"])
    cli = parser.parse_args()
    lock = json.loads((AREA / "configs/VALIDATION_PROTOCOL_LOCK.json").read_text())
    if lock["status"] != "LOCKED_VALIDATION_ONLY":
        raise RuntimeError("protocol not locked for validation only")
    if sha(AREA / "manifests/SPLIT_LOCK.json") != lock["split_lock_sha256"]:
        raise RuntimeError("split lock hash changed")
    for rel, expected in lock["source_hashes"].items():
        if sha(AREA / "source_snapshot" / rel) != expected:
            raise RuntimeError(f"locked source changed: {rel}")
    name = f"HCTF-GD-GraphLocal-v1_{cli.domain}_s{cli.seed}_{cli.variant}"
    config = json.loads((AREA / "configs" / f"{name}.json").read_text())
    if config["run_name"] != name or "--skip-final-test" not in config["command"]:
        raise RuntimeError("run configuration or Test guard mismatch")
    warm = Path(config["command"][config["command"].index("--init-ckpt")+1])
    manifest = AREA / "manifests" / f"{cli.domain}_s{cli.seed}_size_ood.csv"
    if sha(warm) != config["warm_checkpoint_sha256"] or sha(manifest) != config["split_manifest_sha256"]:
        raise RuntimeError("warm checkpoint or split manifest changed")
    best = AREA / "checkpoints" / f"{name}_best.pt"
    last = AREA / "checkpoints" / f"{name}_last.pt"
    status = AREA / "logs" / f"{name}_status.json"
    test_csv = AREA / "metrics" / f"{name}_TEST_RESERVED.csv"
    if any(path.exists() for path in (best, last, status, test_csv)):
        raise RuntimeError("run artifacts already exist; refusing overwrite/retry")
    started = now()
    status.write_text(json.dumps({"run_name": name, "status": "STARTED",
                                  "start_utc": started, "command_config_sha256": sha(AREA / "configs" / f"{name}.json"),
                                  "split_manifest_sha256": sha(manifest),
                                  "warm_checkpoint_sha256": sha(warm)}, indent=2), encoding="utf-8")
    log = AREA / "logs" / f"{name}.log"
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        result = subprocess.run(config["command"], cwd=ROOT,
                                stdout=fh, stderr=subprocess.STDOUT, check=False,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    ended = now()
    history = AREA / "logs" / f"{name}_history.csv"
    completed = {"run_name": name, "status": "VALIDATION_COMPLETE" if result.returncode == 0 and best.exists() and history.exists() and not test_csv.exists() else "FAILED",
                 "exit_code": result.returncode, "start_utc": started, "end_utc": ended,
                 "best_checkpoint_sha256": sha(best) if best.exists() else None,
                 "history_sha256": sha(history) if history.exists() else None,
                 "test_csv_exists": test_csv.exists(),
                 "log_path": str(log), "config_path": str(AREA / "configs" / f"{name}.json")}
    status.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(json.dumps(completed, indent=2))
    if completed["status"] != "VALIDATION_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
