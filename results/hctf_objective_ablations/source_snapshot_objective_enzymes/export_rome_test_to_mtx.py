# export_rome_test_to_mtx.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch

from deepgd.datasets import RomeDataset
from sc_deepgd_moe_gps_ecr import get_split_data


def write_mtx(data, out_path):
    edge_index = data.edge_index.cpu().long()
    num_nodes = int(data.num_nodes)

    edges = set()
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        a, b = min(int(u), int(v)), max(int(u), int(v))
        edges.add((a, b))

    edges = sorted(edges)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{num_nodes} {num_nodes} {len(edges)}\n")
        for u, v in edges:
            # MatrixMarket 是 1-based index
            f.write(f"{u + 1} {v + 1}\n")


def main():
    index_file = "assets/rome_index.txt"
    out_dir = "nnpnet_rome_test_mtx"

    os.makedirs(out_dir, exist_ok=True)

    index = pd.read_csv(index_file, header=None)[0]
    dataset = RomeDataset(index=index)
    datalist = list(dataset)

    test_data = get_split_data(datalist, "test")

    print(f"[INFO] test graphs = {len(test_data)}")
    print(f"[INFO] output dir = {out_dir}")

    for i, data in enumerate(test_data):
        out_path = os.path.join(out_dir, f"rome_test_{i:04d}.mtx")
        write_mtx(data, out_path)

    print("[INFO] done.")


if __name__ == "__main__":
    main()