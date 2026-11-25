"""No-retry serial coordinator for the nine historical strict factorial cohorts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_v2/factorial_ablation"
EVALUATOR = BASE / "evaluate_factorial_validation.py"
RUNS = [(domain, seed) for domain in ("enzymes", "fb15k237", "sbm")
        for seed in (17, 42, 73)]
LOG = BASE / "logs/factorial_cohort.jsonl"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(domain: str, seed: int) -> bool:
    out = BASE / "validation_graphwise" / f"{domain}_s{seed}"
    protocol = out / "PROTOCOL.json"
    if not protocol.exists():
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"incomplete cohort exists; no automatic retry: {out}")
        return False
    p = json.loads(protocol.read_text(encoding="utf-8"))
    if p["status"] != "VALIDATION_FACTORIAL_COMPLETE" or p["domain"] != domain or p["seed"] != seed:
        raise RuntimeError(f"invalid existing protocol: {out}")
    if p["evaluation_script_sha256"] != sha(EVALUATOR):
        raise RuntimeError(f"evaluator source changed since cohort: {out}")
    for filename, key in (("graphwise_factorial_validation.csv", "graphwise_csv_sha256"),
                          ("all_on_equivalence.csv", "equivalence_csv_sha256"),
                          ("coordinates.npz", "coordinates_npz_sha256")):
        if sha(out / filename) != p[key]:
            raise RuntimeError(f"output SHA mismatch: {out / filename}")
    with (out / "graphwise_factorial_validation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != p["rows"] or len(rows) != 8 * p["graphs"]:
        raise RuntimeError(f"row count mismatch: {out}")
    if p["max_all_on_coordinate_abs_diff"] > 1e-6 or p["max_all_on_hidden_abs_diff"] > 1e-6:
        raise RuntimeError(f"all-on difference outside lock: {out}")
    return True


def record(payload: dict) -> None:
    payload["utc"] = datetime.now(timezone.utc).isoformat()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(payload, flush=True)


def main() -> None:
    for domain, seed in RUNS:
        if check(domain, seed):
            record({"domain": domain, "seed": seed, "status": "VERIFIED_EXISTING"})
            continue
        command = [sys.executable, str(EVALUATOR), "--domain", domain, "--seed", str(seed), "--device", "cuda"]
        record({"domain": domain, "seed": seed, "status": "STARTED", "command": command})
        log_path = BASE / "logs" / f"factorial_{domain}_s{seed}.log"
        with log_path.open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code or not check(domain, seed):
            record({"domain": domain, "seed": seed, "status": "FAILED", "exit_code": code,
                    "log_path": str(log_path)})
            raise RuntimeError(f"factorial cohort stopped at {domain} seed {seed}")
        record({"domain": domain, "seed": seed, "status": "COMPLETE", "log_path": str(log_path)})


if __name__ == "__main__":
    main()
