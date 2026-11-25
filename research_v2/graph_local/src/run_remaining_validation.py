"""Run the locked preliminary cohort serially, stopping on first failure.

The executor rechecks source/split/checkpoint hashes and skips final Test for
each run. This orchestrator never changes a configuration or retries a job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    configs = sorted((AREA / "configs").glob("HCTF-GD-GraphLocal-v1_*.json"))
    for number, path in enumerate(configs, 1):
        cfg = json.loads(path.read_text())
        status_path = AREA / "logs" / f"{cfg['run_name']}_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") != "VALIDATION_COMPLETE":
                raise RuntimeError(f"refusing to retry or skip incomplete job: {status_path}")
            print(f"[{now()}] {number}/{len(configs)} already complete: {cfg['run_name']}", flush=True)
            continue
        cmd = [sys.executable, str(AREA / "src/execute_validation.py"),
               "--domain", cfg["domain"], "--seed", str(cfg["seed"]),
               "--variant", cfg["variant"]]
        print(f"[{now()}] {number}/{len(configs)} starting: {cfg['run_name']}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(f"stopped on failed job {cfg['run_name']}: {result.returncode}")
    print(f"[{now()}] all {len(configs)} locked Validation-only runs complete", flush=True)


if __name__ == "__main__":
    main()
