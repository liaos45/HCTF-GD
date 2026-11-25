from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_cser_spectral_mog_end2end.py"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

METRIC_COLUMNS = [
    "batch_stress",
    "edge_crossing",
    "edge_crossing_norm",
    "edge_length_variance",
    "edge_length_cv",
    "node_occlusion_rate",
    "community_compactness",
    "community_separation",
    "community_sep_comp_ratio",
    "community_silhouette",
]

MODERN_COMMON = [
    "--use-cser",
    "--cser-mode",
    "same_spectral",
    "--spectral-dim",
    "6",
    "--use-caid",
    "--use-cpsg",
    "--use-macpsg",
    "--rescale-before-refiner",
    "--select-by",
    "val_refined_stress",
    "--lambda-stress",
    "0.95",
    "--lambda-intra",
    "0.16",
    "--lambda-inter",
    "0.08",
    "--lambda-pair",
    "0.005",
    "--lambda-caid",
    "0.06",
    "--caid-intra-shrink",
    "0.035",
    "--caid-adj-expand",
    "0.01",
    "--caid-far-expand",
    "0.035",
    "--caid-min-ratio",
    "0.96",
    "--caid-max-ratio",
    "1.07",
    "--caid-warmup-epochs",
    "4",
    "--lambda-cpsg-proto",
    "0.035",
    "--lambda-cpsg-sil",
    "0.14",
    "--lambda-cpsg-sep",
    "0.18",
    "--cpsg-sil-margin",
    "0.24",
    "--cpsg-adj-margin",
    "0.48",
    "--cpsg-far-margin",
    "1.18",
    "--cpsg-core-gamma",
    "1.0",
    "--cpsg-boundary-floor",
    "0.52",
    "--cpsg-warmup-epochs",
    "4",
    "--macpsg-sep-scale-floor",
    "1.0",
    "--lambda-pmds-anchor",
    "0.025",
    "--pmds-anchor-start-epoch",
    "1",
    "--pmds-anchor-warmup-epochs",
    "4",
    "--lambda-crossing",
    "0.025",
    "--crossing-start-epoch",
    "1",
    "--crossing-warmup-epochs",
    "4",
    "--lambda-train-silhouette",
    "0.10",
    "--train-silhouette-margin",
    "0.42",
    "--train-silhouette-start-epoch",
    "1",
    "--train-silhouette-warmup-epochs",
    "4",
    "--weight-decay",
    "0.000003",
    "--grad-clip",
    "0.4",
]

LEGACY_COMMON = [
    "--use-cser",
    "--cser-mode",
    "same",
    "--use-caid",
    "--use-cpsg",
    "--use-macpsg",
    "--select-by",
    "val_refined_stress",
    "--lambda-intra",
    "0.85",
    "--lambda-inter",
    "0.65",
    "--lambda-pair",
    "0.08",
    "--lambda-caid",
    "0.25",
    "--caid-intra-shrink",
    "0.12",
    "--caid-adj-expand",
    "0.04",
    "--caid-far-expand",
    "0.18",
    "--caid-min-ratio",
    "0.86",
    "--caid-max-ratio",
    "1.25",
    "--caid-warmup-epochs",
    "4",
    "--lambda-cpsg-proto",
    "0.12",
    "--lambda-cpsg-sil",
    "0.45",
    "--lambda-cpsg-sep",
    "0.85",
    "--cpsg-sil-margin",
    "0.35",
    "--cpsg-adj-margin",
    "0.75",
    "--cpsg-far-margin",
    "2.05",
    "--cpsg-core-gamma",
    "1.5",
    "--cpsg-boundary-floor",
    "0.25",
    "--cpsg-warmup-epochs",
    "4",
    "--macpsg-sep-scale-floor",
    "1.0",
    "--weight-decay",
    "0.000001",
    "--grad-clip",
    "1.0",
]

DATASETS = {
    "sbm": {
        "checkpoint": "sbm_topocser_v1_best.pt",
        "batch_size": 8,
        "lr": "0.000025",
        "crossing_pairs": 1536,
        "flags": MODERN_COMMON + ["--use-topocser"],
    },
    "enzymes": {
        "checkpoint": "enzymes_topocser_v1_best.pt",
        "batch_size": 8,
        "lr": "0.000025",
        "crossing_pairs": 1536,
        "flags": MODERN_COMMON + ["--use-topocser"],
    },
    "proteins": {
        "checkpoint": "proteins_topocser_v1_best.pt",
        "batch_size": 4,
        "lr": "0.000025",
        "crossing_pairs": 768,
        "flags": MODERN_COMMON + ["--use-topocser"],
    },
    "wn18rr": {
        "checkpoint": "wn18rr_spectral_noadaspec_v1_best.pt",
        "batch_size": 8,
        "lr": "0.000018",
        "crossing_pairs": 1536,
        "flags": MODERN_COMMON,
    },
    "fb15k237": {
        "checkpoint": "fb15k237_spectral_noadaspec_v1_best.pt",
        "batch_size": 8,
        "lr": "0.000018",
        "crossing_pairs": 1536,
        "flags": MODERN_COMMON,
    },
    "rome": {
        "checkpoint": "caer_caid_macpsg_commstrong_best.pt",
        "batch_size": 16,
        "lr": "0.0002",
        "crossing_pairs": 512,
        "flags": LEGACY_COMMON
        + [
            "--lambda-stress",
            "0.30",
            "--lambda-intra",
            "0.90",
            "--lambda-inter",
            "0.70",
            "--lambda-caid",
            "0.25",
            "--lambda-cpsg-sep",
            "0.80",
        ],
    },
    "north": {
        "checkpoint": "north_caer_caid_macpsg_best.pt",
        "batch_size": 16,
        "lr": "0.0002",
        "crossing_pairs": 512,
        "flags": LEGACY_COMMON + ["--lambda-stress", "0.32"],
    },
    "imdb_binary": {
        "checkpoint": "imdb_binary_caer_caid_macpsg_best.pt",
        "batch_size": 16,
        "lr": "0.0002",
        "crossing_pairs": 512,
        "flags": LEGACY_COMMON + ["--lambda-stress", "0.32"],
    },
}


def write_status(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "hctf_all8_pilot_status.csv",
        index=False,
        encoding="utf-8-sig",
    )


def collect_summary(status_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for status in status_rows:
        if status["status"] != "SUCCESS":
            continue
        eval_path = ROOT / status["eval_csv"]
        if not eval_path.exists():
            continue
        frame = pd.read_csv(eval_path)
        available = [column for column in METRIC_COLUMNS if column in frame]
        row = {
            "dataset": status["dataset"],
            "variant": status["variant"],
            "graphs": len(frame),
            "elapsed_seconds": status["elapsed_seconds"],
        }
        row.update(frame[available].mean().to_dict())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULTS_DIR / "hctf_all8_pilot_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deltas = []
    if not summary.empty:
        for dataset, group in summary.groupby("dataset"):
            indexed = group.set_index("variant")
            if not {"baseline", "hctf"}.issubset(indexed.index):
                continue
            row = {"dataset": dataset}
            for metric in METRIC_COLUMNS:
                if metric not in indexed:
                    continue
                baseline = float(indexed.loc["baseline", metric])
                hctf = float(indexed.loc["hctf", metric])
                row[f"{metric}_baseline"] = baseline
                row[f"{metric}_hctf"] = hctf
                row[f"{metric}_delta_pct"] = (
                    (hctf - baseline) / abs(baseline) * 100.0
                    if baseline != 0.0
                    else float("nan")
                )
            deltas.append(row)
    pd.DataFrame(deltas).to_csv(
        RESULTS_DIR / "hctf_all8_pilot_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def build_command(
    dataset: str,
    variant: str,
    epochs: int,
    max_train_batches: int,
    max_val_batches: int,
) -> tuple[list[str], str, str]:
    spec = DATASETS[dataset]
    run_name = f"{dataset}_hctf_all8_{variant}_pilot"
    eval_csv = f"results/{run_name}_test.csv"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--mode",
        "train",
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(spec["batch_size"]),
        "--max-train-batches",
        str(max_train_batches),
        "--max-val-batches",
        str(max_val_batches),
        "--max-test-batches",
        "-1",
        "--init-ckpt",
        f"checkpoints/{spec['checkpoint']}",
        "--min-select-epoch",
        "1",
        "--crossing-max-pairs",
        str(spec["crossing_pairs"]),
        "--deepgd-lr",
        str(spec["lr"]),
        "--memory-cleanup-every",
        "1",
        "--no-save-optimizer",
        "--seed",
        "42",
        "--run-name",
        run_name,
        "--eval-csv",
        eval_csv,
    ]
    command.extend(spec["flags"])
    if variant == "hctf":
        command.extend(
            [
                "--use-hctf",
                "--hctf-num-trees",
                "3",
                "--hctf-hidden-dim",
                "16",
                "--hctf-dropout",
                "0.05",
                "--hctf-scale",
                "0.20",
                "--hctf-block-stride",
                "1",
            ]
        )
    return command, run_name, eval_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=["baseline", "hctf"], default=["baseline", "hctf"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=12)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    status_rows: list[dict] = []
    failures = 0

    for dataset in args.datasets:
        checkpoint = ROOT / "checkpoints" / DATASETS[dataset]["checkpoint"]
        if not checkpoint.exists():
            for variant in args.variants:
                status_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "status": "MISSING_CHECKPOINT",
                        "return_code": -1,
                        "elapsed_seconds": 0.0,
                        "eval_csv": "",
                        "log": "",
                    }
                )
            failures += 1
            write_status(status_rows)
            print(f"[MISSING] {dataset}: {checkpoint}", flush=True)
            continue

        for variant in args.variants:
            command, run_name, eval_csv = build_command(
                dataset,
                variant,
                args.epochs,
                args.max_train_batches,
                args.max_val_batches,
            )
            eval_path = ROOT / eval_csv
            log_relative = f"logs/{run_name}.log"
            log_path = ROOT / log_relative
            if args.skip_completed and eval_path.exists():
                status = "SKIPPED_EXISTING"
                return_code = 0
                elapsed = 0.0
                print(f"[SKIP] {dataset}/{variant}: {eval_path}", flush=True)
            else:
                print(f"[RUN] {dataset}/{variant}", flush=True)
                started = time.perf_counter()
                with log_path.open("wb") as log_file:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                elapsed = time.perf_counter() - started
                return_code = int(completed.returncode)
                status = "SUCCESS" if return_code == 0 and eval_path.exists() else "FAILED"
                print(
                    f"[{status}] {dataset}/{variant}: {elapsed:.1f}s, log={log_relative}",
                    flush=True,
                )
                if status == "FAILED":
                    failures += 1
            status_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "status": status,
                    "return_code": return_code,
                    "elapsed_seconds": round(elapsed, 3),
                    "eval_csv": eval_csv,
                    "log": log_relative,
                }
            )
            write_status(status_rows)
            collect_summary(status_rows)

    summary = collect_summary(status_rows)
    print(
        f"[DONE] successful result rows={len(summary)}, failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
