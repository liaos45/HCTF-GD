"""Recompute point estimates for the two external-baseline tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from verify import DOMAINS, ROOT, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = ROOT / "results"
    rows = []
    for domain in DOMAINS:
        for seed in SEEDS:
            for variant, method in [("baseline", "Backbone"), ("full_nodrop", "Full"),
                                    ("node_only_nodrop", "Node")]:
                data = pd.read_csv(root / "hctf_submission_main" / f"submission_{domain}_{variant}_s{seed}_test.csv")
                rows.append((domain, seed, method, data.graph_stress.mean()))
            original = pd.read_csv(root / "original_deepgd_baselines" / f"original_deepgd_{domain}_s{seed}.csv")
            rows.append((domain, seed, "DeepGD", original.graph_stress.mean()))
            classical = pd.read_csv(root / "hctf_classical_baselines" / f"{domain}_all_s{seed}.csv")
            for method, subset in classical.groupby("method"):
                rows.append((domain, seed, "FA2" if method == "FORCEATLAS2" else method,
                             subset.graph_stress.mean()))
    seed_means = pd.DataFrame(rows, columns=["dataset", "seed", "method", "mean_stress"])
    saved = pd.read_csv(ROOT / "reference/evidence/raw_baseline_seed_means.csv")
    keys = ["dataset", "seed", "method"]
    pd.testing.assert_frame_equal(seed_means.set_index(keys).sort_index(), saved.set_index(keys).sort_index(),
                                  rtol=1e-10, atol=1e-12)
    seed_means.groupby(["dataset", "method"]).mean_stress.mean().unstack().to_csv(args.out / "table3.csv")

    new = []
    for path in ["autofdp_2026/autofdp_all_graph_metrics.csv", "nnpnet_2026_static/nnpnet_all_graph_metrics.csv"]:
        data = pd.read_csv(root / path)
        assert set(data.seed) == {42}
        for (domain, method), subset in data.groupby(["dataset", "method"]):
            base = pd.read_csv(root / "hctf_submission_main" / f"submission_{domain}_baseline_s42_test.csv")
            dims = ["num_nodes", "num_edges"]
            np.testing.assert_array_equal(subset.sort_values("test_graph_id")[dims].to_numpy(),
                                          base.sort_values(["batch_id", "local_graph_id"])[dims].to_numpy())
            new.append((domain, method, subset.graph_stress.mean()))
    recent = pd.DataFrame(new, columns=["dataset", "method", "mean_stress"])
    saved = pd.read_csv(ROOT / "reference/evidence/recent_baseline_means.csv").rename(columns={"stress": "mean_stress"})
    keys = ["dataset", "method"]
    pd.testing.assert_frame_equal(recent.set_index(keys).sort_index(),
                                  saved.drop(columns="graphs").set_index(keys).sort_index(),
                                  rtol=1e-10, atol=1e-12)
    hctf = seed_means[seed_means.seed.eq(42) & seed_means.method.isin(["Full", "Node"])].drop(columns="seed")
    pd.concat([recent, hctf]).pivot(index="dataset", columns="method", values="mean_stress").to_csv(
        args.out / "table4.csv"
    )
    print("Recomputed Table 3 and Table 4 point estimates from archived inputs.")


if __name__ == "__main__":
    main()
