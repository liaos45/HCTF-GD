# evaluate_egraph_omega_layouts.py
# -*- coding: utf-8 -*-

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from tqdm import tqdm

from deepgd.metrics import Stress
from evaluate_classical_layouts import (
    dataset_class_from_name,
    evaluate_batch_layout,
    split_graph_dataset,
)


def configure_dataset_assets(args):
    if args.dataset == "north":
        args.index_file = args.index_file or "assets/north_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/north_pmds.npy"
    elif args.dataset == "sbm":
        args.index_file = args.index_file or "assets/sbm_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/sbm_pmds.npy"
    elif args.dataset == "fb15k237":
        args.index_file = args.index_file or "assets/fb15k237_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/fb15k237_pmds.npy"
    elif args.dataset == "wn18rr":
        args.index_file = args.index_file or "assets/wn18rr_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/wn18rr_pmds.npy"
    elif args.dataset == "imdb_binary":
        args.index_file = args.index_file or "assets/imdb_binary_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/imdb_binary_pmds.npy"
    elif args.dataset == "proteins":
        args.index_file = args.index_file or "assets/proteins_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/proteins_pmds.npy"
    elif args.dataset == "enzymes":
        args.index_file = args.index_file or "assets/enzymes_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/enzymes_pmds.npy"
    elif args.dataset == "rome":
        args.index_file = args.index_file or "assets/rome_index.txt"
        args.layouts_file = args.layouts_file or "assets/layouts/pmds.npy"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def import_egraph_or_explain():
    try:
        import egraph as eg  # type: ignore

        return eg
    except Exception as exc:
        msg = f"""
[ERROR] Could not import egraph Python bindings: {exc}

Install the official egraph-rs Python binding first:

  winget install Rustlang.Rustup
  # open a new terminal after Rust installation
  py -3.13 -m pip install maturin
  cd /d E:\\Projiect6\\egraph-rs\\crates\\python
  py -3.13 -m maturin develop --release
  py -3.13 -c "import egraph as eg; print(eg)"

Then rerun this script from E:\\Projiect6\\DeepGD-main.
"""
        raise RuntimeError(msg) from exc


def pyg_data_to_egraph(data, eg):
    graph = eg.Graph()
    num_nodes = int(data.num_nodes)
    nodes = [graph.add_node(i) for i in range(num_nodes)]

    edge_index = data.edge_index.long().detach().cpu().numpy()
    seen = set()
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        graph.add_edge(nodes[a], nodes[b], None)

    return graph


def normalize_pos(pos):
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        pos = np.zeros((pos.shape[0], 2), dtype=np.float32)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    pos = pos - pos.mean(axis=0, keepdims=True)
    scale = float(np.std(pos))
    if scale > 1e-9:
        pos = pos / scale
    return pos.astype(np.float32)


def drawing_to_numpy(drawing, num_nodes):
    pos = np.zeros((num_nodes, 2), dtype=np.float32)
    for i in range(num_nodes):
        x = drawing.x(i)
        y = drawing.y(i)
        pos[i, 0] = 0.0 if x is None else float(x)
        pos[i, 1] = 0.0 if y is None else float(y)
    return normalize_pos(pos)


def run_egraph_layout(data, eg, args, method):
    num_nodes = int(data.num_nodes)
    if num_nodes <= 1:
        return np.zeros((num_nodes, 2), dtype=np.float32)

    graph = pyg_data_to_egraph(data, eg)
    if graph.edge_count() == 0:
        return normalize_pos(data.pos.detach().cpu().numpy())

    rng = eg.Rng.seed_from(int(args.seed))

    rdmds = (
        eg.RdMds()
        .d(int(args.embedding_dim))
        .shift(float(args.rd_shift))
        .eigenvalue_max_iterations(int(args.eigenvalue_max_iterations))
        .cg_max_iterations(int(args.cg_max_iterations))
        .eigenvalue_tolerance(float(args.eigenvalue_tolerance))
        .cg_tolerance(float(args.cg_tolerance))
    )
    embedding_obj = rdmds.embedding(graph, lambda edge_idx: float(args.edge_length), rng)
    embedding = normalize_pos(np.asarray(embedding_obj.to_numpy(), dtype=np.float32)[:, :2])

    if method == "rdmds":
        return embedding

    drawing = eg.DrawingEuclidean2d.from_array2(graph, eg.Array2(embedding.astype(np.float64)))
    sgd = (
        eg.Omega()
        .k(int(args.omega_k))
        .min_dist(float(args.omega_min_dist))
        .build(graph, embedding_obj, rng)
    )
    scheduler = sgd.scheduler(int(args.omega_steps), float(args.omega_eta))

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)

    scheduler.run(step)
    return drawing_to_numpy(drawing, num_nodes)


def load_test_graphs(args):
    configure_dataset_assets(args)
    print(f"[INFO] loading {args.dataset} dataset...")
    dataset_cls = dataset_class_from_name(args.dataset)
    index = pd.read_csv(args.index_file, header=None)[0]
    dataset = dataset_cls(index=index)
    layouts = np.load(args.layouts_file, allow_pickle=True)

    datalist = list(dataset)
    for i, data in enumerate(datalist):
        data.pos = torch.tensor(layouts[i]).float()

    test_list = split_graph_dataset(datalist, args.dataset, args.seed)
    if args.max_test_graphs > 0:
        test_list = test_list[: args.max_test_graphs]
    print(f"[INFO] test graphs = {len(test_list)}")
    return test_list


def method_names(args):
    if args.method == "both":
        return ["rdmds", "omega"]
    return [args.method]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="sbm",
        choices=[
            "rome",
            "north",
            "sbm",
            "fb15k237",
            "wn18rr",
            "imdb_binary",
            "proteins",
            "enzymes",
        ],
    )
    parser.add_argument("--index-file", type=str, default=None)
    parser.add_argument("--layouts-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-test-graphs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="omega", choices=["rdmds", "omega", "both"])
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--edge-length", type=float, default=1.0)
    parser.add_argument("--rd-shift", type=float, default=1e-3)
    parser.add_argument("--eigenvalue-max-iterations", type=int, default=500)
    parser.add_argument("--cg-max-iterations", type=int, default=50)
    parser.add_argument("--eigenvalue-tolerance", type=float, default=1e-2)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
    parser.add_argument("--omega-k", type=int, default=15)
    parser.add_argument("--omega-min-dist", type=float, default=1e-3)
    parser.add_argument("--omega-steps", type=int, default=80)
    parser.add_argument("--omega-eta", type=float, default=0.1)
    parser.add_argument("--fallback", type=str, default="pmds", choices=["pmds", "skip"])
    parser.add_argument("--output-csv", type=str, default="results/egraph_omega_eval.csv")
    args = parser.parse_args()

    eg = import_egraph_or_explain()
    test_list = load_test_graphs(args)

    stress_metric = Stress()
    all_rows = []
    failure_rows = []

    for method in method_names(args):
        display_name = "EGRAPH-RDMDS" if method == "rdmds" else "EGRAPH-OMEGA"
        generated = []
        print(f"\n========== Running {display_name} ==========")

        for graph_id, data in enumerate(tqdm(test_list, desc=f"Generate {display_name}")):
            start = time.time()
            failed = False
            try:
                pos_np = run_egraph_layout(data, eg, args, method)
            except Exception as exc:
                failed = True
                failure_rows.append(
                    {
                        "method": display_name,
                        "graph_id": graph_id,
                        "error": repr(exc),
                    }
                )
                if args.fallback == "skip":
                    continue
                pos_np = normalize_pos(data.pos.detach().cpu().numpy())

            new_data = data.clone()
            new_data.pos = torch.tensor(pos_np, dtype=torch.float32)
            new_data.layout_time = float(time.time() - start)
            new_data.egraph_failed = int(failed)
            generated.append(new_data)

        loader = pyg.loader.DataLoader(generated, batch_size=args.batch_size, shuffle=False)
        method_rows = []
        stress_values = []
        for batch_id, batch in enumerate(tqdm(loader, desc=f"Evaluate {display_name}")):
            rows, stress_value = evaluate_batch_layout(
                batch=batch,
                pos=batch.pos,
                stress_metric=stress_metric,
                method_name=display_name,
                batch_id=batch_id,
                infer_time=0.0,
            )
            method_rows.extend(rows)
            stress_values.append(float(stress_value))

        all_rows.extend(method_rows)
        df_method = pd.DataFrame(method_rows)
        print(f"\n[{display_name}] Mean Stress: {np.mean(stress_values):.6f}")
        print(f"[{display_name}] Edge Crossing: {df_method['edge_crossing'].mean():.6f}")
        print(f"[{display_name}] Edge Length CV: {df_method['edge_length_cv'].mean():.6f}")
        print(f"[{display_name}] Sep/Comp Ratio: {df_method['community_sep_comp_ratio'].mean():.6f}")
        print(f"[{display_name}] Silhouette: {df_method['community_silhouette'].mean():.6f}")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    if failure_rows:
        failure_csv = args.output_csv.replace(".csv", "_failures.csv")
        pd.DataFrame(failure_rows).to_csv(failure_csv, index=False, encoding="utf-8-sig")
        print(f"[WARN] failures = {len(failure_rows)}, saved to: {failure_csv}")

    metric_cols = [
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
        "num_communities",
    ]
    print("\n========== EGRAPH Layout Summary ==========")
    print(df.groupby("method")[metric_cols].mean())
    print(f"\n[INFO] saved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
