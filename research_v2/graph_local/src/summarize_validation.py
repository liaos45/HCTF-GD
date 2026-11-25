"""Summarize every completed locked Validation run, including adverse logs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"


def main() -> None:
    rows = []
    for config_path in sorted((AREA / "configs").glob("HCTF-GD-GraphLocal-v1_*.json")):
        cfg = json.loads(config_path.read_text())
        name = cfg["run_name"]
        status_path = AREA / "logs" / f"{name}_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "NOT_STARTED"}
        row = {"run_name": name, "domain": cfg["domain"], "seed": cfg["seed"],
               "variant": cfg["variant"], "status": status["status"],
               "best_checkpoint_sha256": status.get("best_checkpoint_sha256", ""),
               "test_csv_exists": status.get("test_csv_exists", False)}
        history_path = AREA / "logs" / f"{name}_history.csv"
        if history_path.exists():
            frame = pd.read_csv(history_path)
            required = ("epoch", "train_loss", "val_loss", "val_refined_stress")
            if any(col not in frame for col in required):
                raise RuntimeError(f"missing required Validation column in {history_path}")
            for col in required[1:]:
                if not frame[col].map(math.isfinite).all():
                    row["finite_all_required"] = False
                    break
            else:
                row["finite_all_required"] = True
            idx = frame["val_refined_stress"].idxmin()
            row.update({"epochs_logged": len(frame),
                        "best_val_epoch": int(frame.loc[idx, "epoch"]),
                        "best_val_refined_stress": float(frame.loc[idx, "val_refined_stress"]),
                        "first_val_refined_stress": float(frame.iloc[0]["val_refined_stress"]),
                        "last_val_refined_stress": float(frame.iloc[-1]["val_refined_stress"]),
                        "train_loss_median": float(frame["train_loss"].median()),
                        "train_loss_max": float(frame["train_loss"].max()),
                        "train_loss_spike_ratio": float(frame["train_loss"].max() / max(abs(frame["train_loss"].median()), 1e-12)),
                        "gate_first": float(frame.iloc[0].get("hctf_structure_gate", float("nan"))),
                        "gate_last": float(frame.iloc[-1].get("hctf_structure_gate", float("nan")))})
        rows.append(row)
    out = AREA / "metrics/validation_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(row['status']=='VALIDATION_COMPLETE' for row in rows)}/{len(rows)} complete; wrote {out}")


if __name__ == "__main__":
    main()
