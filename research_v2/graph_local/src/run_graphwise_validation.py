"""Serial, no-retry coordinator for locked GraphLocal-v1 Validation inference."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research_v2/graph_local"
EVALUATOR = Path(__file__).with_name("evaluate_validation_graphwise.py")
LOG = BASE / "logs/graphwise_validation_cohort.jsonl"
RUNS = [(d, s) for d in ("enzymes", "fb15k237", "sbm") for s in (17, 42, 73)]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_existing(domain: str, seed: int) -> bool:
    out = BASE / "validation_graphwise" / f"{domain}_s{seed}"
    protocol = out / "PROTOCOL.json"
    if not protocol.exists():
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"incomplete output exists; refusing automatic retry: {out}")
        return False
    p = json.loads(protocol.read_text(encoding="utf-8"))
    if p["status"] != "VALIDATION_GRAPHWISE_COMPLETE" or p["domain"] != domain or p["seed"] != seed:
        raise RuntimeError(f"invalid existing protocol: {out}")
    for filename, key in (("graphwise_validation.csv", "graphwise_csv_sha256"),
                          ("coordinates.npz", "coordinates_npz_sha256")):
        if sha(out / filename) != p[key]:
            raise RuntimeError(f"existing output hash mismatch: {out / filename}")
    if p["evaluation_script_sha256"] != sha(EVALUATOR):
        raise RuntimeError(f"evaluator source differs from existing output: {out}")
    with (out / "graphwise_validation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != p["rows"] or len(rows) != 4 * p["graphs"]:
        raise RuntimeError(f"existing row count mismatch: {out}")
    return True


def record(item: dict) -> None:
    item["utc"] = datetime.now(timezone.utc).isoformat()
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(item, flush=True)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for domain, seed in RUNS:
        if check_existing(domain, seed):
            record({"domain": domain, "seed": seed, "status": "VERIFIED_EXISTING"})
            continue
        command = [sys.executable, str(EVALUATOR), "--domain", domain,
                   "--seed", str(seed), "--device", "cuda"]
        record({"domain": domain, "seed": seed, "status": "STARTED", "command": command})
        log_path = BASE / "logs" / f"graphwise_{domain}_s{seed}.log"
        with log_path.open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code or not check_existing(domain, seed):
            record({"domain": domain, "seed": seed, "status": "FAILED",
                    "exit_code": code, "log_path": str(log_path)})
            raise RuntimeError(f"cohort stopped at {domain} seed {seed}")
        record({"domain": domain, "seed": seed, "status": "COMPLETE",
                "log_path": str(log_path)})


if __name__ == "__main__":
    main()
