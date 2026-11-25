"""Check release integrity and recompute the main archive counts without a model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("enzymes", "fb15k237", "imdb_binary", "north", "proteins", "rome", "sbm", "wn18rr")
SEEDS = (17, 42, 73, 101, 137)
VARIANTS = ("baseline", "node_only_nodrop", "full_nodrop")
AUDIT_FIELDS = [
    "batch_stress", "edge_crossing", "edge_crossing_norm", "edge_length_variance",
    "edge_length_cv", "node_occlusion_rate", "community_compactness", "community_separation",
    "community_sep_comp_ratio", "community_silhouette",
]
POST_FIELDS = [
    "angular_resolution_deg", "angular_resolution_ratio", "crossing_angle_deg",
    "crossing_angle_p05_deg", "node_edge_clearance", "node_edge_clearance_p05",
    "neighborhood_preservation_k5",
]


def verify_files():
    with (ROOT / "FILE_MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row["path"]
        assert path.stat().st_size == int(row["bytes"]), row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
    assert len(list((ROOT / "paper_figures").glob("*.pdf"))) == 10
    return len(rows)


def recount():
    results = ROOT / "results"
    records = 0
    selection = {v: {"total": 0, "accepted": 0, "fallback": 0} for v in VARIANTS[1:]}
    batches = {v: [] for v in VARIANTS}
    for domain in DOMAINS:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = results / "hctf_submission_main" / f"submission_{domain}_{variant}_s{seed}_test.csv"
                main = pd.read_csv(path)
                assert not main.duplicated(["batch_id", "local_graph_id"]).any()
                records += len(main)
                grouped = main.groupby("batch_id").batch_infer_time
                assert grouped.nunique().eq(1).all()
                batches[variant].extend(1000 * grouped.first() / grouped.size())
                if variant == "baseline":
                    continue
                path = results / "hctf_safe_blend_paired_holdout" / f"{domain}_{variant}_s{seed}"
                saved = pd.read_csv(path / f"{domain}_selection.csv")
                assert saved.accepted.eq(saved.selected_alpha.gt(0)).all()
                assert set(saved.selected_alpha).issubset({0., .0625, .125, .25, .5, .75, 1.})
                frame = pd.read_csv(path / f"{domain}_eval.csv").set_index(["batch_id", "graph_id"])
                base = frame[frame.method.eq("Current-Model-Baseline")]
                final = frame[frame.method.eq("HCTF-SafeBlend")]
                fallback = saved.loc[~saved.accepted].set_index(["batch_id", "graph_id"]).index
                fields = AUDIT_FIELDS + POST_FIELDS
                pd.testing.assert_frame_equal(base.loc[fallback, fields], final.loc[fallback, fields],
                                              check_exact=True)
                selection[variant]["total"] += len(saved)
                selection[variant]["accepted"] += int(saved.accepted.sum())
                selection[variant]["fallback"] += len(fallback)
    assert records == 29175
    assert selection["node_only_nodrop"] == {"total": 9725, "accepted": 870, "fallback": 8855}
    assert selection["full_nodrop"] == {"total": 9725, "accepted": 970, "fallback": 8755}
    latencies = {key: float(np.median(value)) for key, value in batches.items()}
    assert sum(map(len, batches.values())) == 2745
    assert [round(latencies[key], 1) for key in VARIANTS] == [23.0, 27.5, 29.8]
    return records, selection, latencies


def main():
    files = verify_files()
    records, selection, medians = recount()
    print(json.dumps({"verified_files": files, "legacy_records": records, "safeblend": selection,
                      "batch_derived_ms_per_graph_medians": medians}, indent=2))


if __name__ == "__main__":
    main()
