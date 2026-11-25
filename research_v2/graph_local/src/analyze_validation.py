"""Report locked Phase 2 Validation summaries; no Test file is read."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "research_v2/graph_local"
OUT = ROOT / "research_v2/reports/graph_local_validation_results.md"


def main() -> None:
    source = AREA / "metrics/validation_summary.csv"
    frame = pd.read_csv(source)
    if len(frame) != 36 or not frame.status.eq("VALIDATION_COMPLETE").all():
        raise RuntimeError("all 36 locked Validation runs must finish first")
    if not frame.finite_all_required.all() or frame.test_csv_exists.any():
        raise RuntimeError("nonfinite training/Validation value or unexpected Test CSV")
    p = frame.pivot(index=["domain", "seed"], columns="variant", values="best_val_refined_stress")
    if p.shape != (9, 4) or p.isna().any().any():
        raise RuntimeError("missing or duplicate domain/seed/variant")
    p = p.rename(columns={"A_both": "A", "B_caid_only": "B",
                          "C_macpsg_only": "C", "D_neither": "D"})
    p["A_vs_B_pct"] = 100 * (p.B-p.A)/p.B
    p["A_vs_C_pct"] = 100 * (p.C-p.A)/p.C
    p["A_vs_D_pct"] = 100 * (p.D-p.A)/p.D
    p["caid_main_pct"] = 100 * ((p.D-p.B)+(p.C-p.A))/(2*p.D)
    p["macpsg_main_pct"] = 100 * ((p.D-p.C)+(p.B-p.A))/(2*p.D)
    p["interaction_pct"] = 100 * (p.B+p.C-p.D-p.A)/p.D
    p.to_csv(AREA / "metrics/validation_factorial_effects.csv")
    header = "| Domain | Seed | A both | B CAID | C MA-CPSG | D neither | A vs B % | A vs C % | A vs D % |"
    lines = ["# GraphLocal-v1 36-run Validation-only analysis", "",
             "**Status:** 36/36 Train/Validation jobs complete; 9/9 four-way checkpoint audits pass; no final Test CSV exists. These are archived validation-selection-scale `val_refined_stress` values (lower is better), not pair-normalized graphwise Test Stress. No graphwise Validation coordinates were archived here, so no graph-cluster bootstrap interval is computed or inferred.",
             "", "## Every domain and seed", "", header,
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (domain, seed), row in p.iterrows():
        lines.append(f"| {domain} | {seed} | {row.A:.6f} | {row.B:.6f} | {row.C:.6f} | {row.D:.6f} | {row.A_vs_B_pct:+.5f} | {row.A_vs_C_pct:+.5f} | {row.A_vs_D_pct:+.5f} |")
    grouped = p.groupby(level=0)[["A_vs_B_pct", "A_vs_C_pct", "A_vs_D_pct", "caid_main_pct", "macpsg_main_pct", "interaction_pct"]].mean()
    lines += ["", "## Descriptive seed means by domain (%)", "",
              "The first three columns compare Full A with the named alternative; positive favors A. Main effects average the component's two conditional contrasts, normalized by D within each domain/seed. A positive interaction means A has a lower cost than the sum-of-parts prediction. All entries are descriptive, not confidence intervals.",
              "", "| Domain | A vs B | A vs C | A vs D | CAID main | MA-CPSG main | Interaction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for domain, row in grouped.iterrows():
        lines.append(f"| {domain} | {row.A_vs_B_pct:+.5f} | {row.A_vs_C_pct:+.5f} | {row.A_vs_D_pct:+.5f} | {row.caid_main_pct:+.5f} | {row.macpsg_main_pct:+.5f} | {row.interaction_pct:+.5f} |")
    mean = grouped.mean()
    lines.append(f"| Equal-domain mean | {mean.A_vs_B_pct:+.5f} | {mean.A_vs_C_pct:+.5f} | {mean.A_vs_D_pct:+.5f} | {mean.caid_main_pct:+.5f} | {mean.macpsg_main_pct:+.5f} | {mean.interaction_pct:+.5f} |")
    best = p[["A", "B", "C", "D"]].idxmin(axis=1).value_counts().to_dict()
    early = frame[frame.epochs_logged < 20]
    lines += ["", "## Integrity, limitations and decision", "",
              f"- Best-setting counts across the nine domain/seed cells: {best}. This is not a seed-selection rule.",
              f"- {len(early)}/36 jobs stopped before epoch 20 under the identical registered early-stopping rule; their recorded epochs are {[(r.domain, int(r.seed), r.variant, int(r.epochs_logged)) for r in early.itertuples(index=False)]}.",
              f"- All required recorded loss/Validation values are finite. The largest train-loss-to-median ratio is {frame.train_loss_spike_ratio.max():.1f}; the PMDS-anchor spikes were also present in the historical strict trainer and are retained as a stability concern.",
              "- The four settings share source snapshot, warm checkpoint, split and architecture per domain/seed. `matched_checkpoint_audit.json` confirms every non-HCTF tensor identical within all nine four-way groups.",
              "- The sign and size of A's Validation difference vary by domain/seed. The current data **do not establish** an independent, stable benefit for CAID, MA-CPSG, or their combination. No claim about graphwise Crossing, community geometry, generalization, historical Legacy superiority, or formal Test is supported by this table.",
              "- The next registered stage is same-checkpoint component-control Validation, followed by new-domain source/split qualification. Formal Test remains closed.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}; best-setting counts={best}")


if __name__ == "__main__":
    main()
