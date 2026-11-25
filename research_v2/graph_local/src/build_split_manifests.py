"""Reproduce archived size ordering from graph metadata, without inference.

Only graph names, sizes, and edge counts are read. No layout, checkpoint, or
evaluation metric is loaded. This is for the already exposed three domains.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research_v2/graph_local/manifests"
DATASETS = {"enzymes": "ENZYMES", "fb15k237": "FB15k237Sampled", "sbm": "SBM"}
SEEDS = (17, 42, 73)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for domain, folder in DATASETS.items():
        processed = ROOT / "datasets" / folder / "processed" / "data.pt"
        index = ROOT / "datasets" / folder / "processed" / "index.txt"
        data, _ = torch.load(processed, weights_only=False, map_location="cpu")
        names = index.read_text(encoding="utf-8").splitlines()
        graph_by_name = {str(graph.graph["name"]): graph for graph in data.G}
        if len(names) != len(graph_by_name) or set(names) != set(graph_by_name):
            raise RuntimeError(f"{domain}: index and processed graph identities differ")
        graph_size = [len(graph_by_name[name]) for name in names]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tie = rng.random(len(names))
            ordered = sorted(range(len(names)), key=lambda i: (graph_size[i], float(tie[i])))
            n = len(ordered)
            train_end = max(1, int(0.70*n))
            val_end = min(max(train_end+1, int(0.85*n)), n-1)
            labels = [None]*n
            for split, indices in (("train", ordered[:train_end]),
                                   ("validation", ordered[train_end:val_end]),
                                   ("test_reserved", ordered[val_end:])):
                for i in indices:
                    labels[i] = split
            rows = []
            for i, name in enumerate(names):
                graph = graph_by_name[name]
                rows.append({"domain": domain, "seed": seed, "graph_id": name,
                             "node_count": len(graph),
                             "edge_count_undirected": graph.to_undirected().number_of_edges(),
                             "source_family": "TU" if domain == "enzymes" else ("KG-sampled" if domain == "fb15k237" else "SBM-generated"),
                             "split": labels[i]})
            path = OUT / f"{domain}_s{seed}_size_ood.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            train_max = max(r["node_count"] for r in rows if r["split"] == "train")
            val_min = min(r["node_count"] for r in rows if r["split"] == "validation")
            test_min = min(r["node_count"] for r in rows if r["split"] == "test_reserved")
            if test_min <= train_max:
                raise RuntimeError(f"{domain} seed {seed}: archived size separation failed")
            summary.append({"domain": domain, "seed": seed, "n_graphs": n,
                            "n_train": train_end, "n_val": val_end-train_end,
                            "n_test_reserved": n-val_end,
                            "train_nodes_max": train_max, "val_nodes_min": val_min,
                            "test_nodes_min": test_min, "manifest_sha256": sha(path),
                            "processed_sha256": sha(processed), "index_sha256": sha(index)})
    (OUT / "SPLIT_LOCK.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Locked {len(summary)} metadata-only split manifests in {OUT}")


if __name__ == "__main__":
    main()
