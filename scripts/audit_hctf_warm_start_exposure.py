"""Reconstruct recorded warm-start splits; do not assume a new seed is unseen.

The audit uses the actual dataset loader and checkpoint args, including ROME's
canonical split. It identifies recorded split exposure, not an unavailable full
ancestry of resumed checkpoints or historical method-development decisions.
"""
import argparse
import gc
import json
import pandas as pd
from evaluate_hctf_safe_blend import ROOT, train, namespace_from_checkpoint
from run_hctf_all8_pilot import DATASETS as CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(CONFIG))
    parser.add_argument("--output-dir", default="results/hctf_warm_start_exposure")
    args = parser.parse_args()
    metadata = pd.read_csv(ROOT/"results/hctf_graph_metadata.csv")
    rows, summary = [], []
    for dataset in args.datasets:
        path = ROOT/"checkpoints"/CONFIG[dataset]["checkpoint"]
        warm_args = namespace_from_checkpoint(path)
        required = ["dataset","seed","index_file","layouts_file","batch_size"]
        if any(not hasattr(warm_args,name) for name in required):
            for seed, subset in metadata[metadata.dataset==dataset].groupby("seed"):
                for row in subset.itertuples(index=False):
                    rows.append({"dataset":dataset,"seed":int(seed),"test_graph_id":row.test_graph_id,
                        "original_graph_index":row.original_graph_index,"graph_name":row.graph_name,
                        "recorded_warm_split":"unknown","warm_checkpoint":str(path.relative_to(ROOT)),"warm_seed":None})
                summary.append({"dataset":dataset,"seed":int(seed),"test_graphs":len(subset),
                    "warm_train_overlap":None,"warm_validation_overlap":None,"warm_test_overlap":None,
                    "warm_train_overlap_fraction":None,"warm_checkpoint":str(path.relative_to(ROOT)),
                    "warm_seed":None,"warm_split_reconstruction_available":False})
            print(f"[EXPOSURE UNKNOWN] {dataset}: legacy checkpoint lacks recorded split arguments",flush=True)
            continue
        train.set_random_seed(warm_args.seed)
        training, validation, test = train.load_graph_dataset(warm_args)
        pools = {name: {str(data.name) for data in items} for name, items in
                 [("train", training), ("validation", validation), ("test", test)]}
        if any(pools[a]&pools[b] for a,b in [("train","validation"),("train","test"),("validation","test")]):
            raise ValueError(f"Non-disjoint recorded warm-start split: {dataset}")
        for seed, subset in metadata[metadata.dataset==dataset].groupby("seed"):
            counts = {name: 0 for name in pools}
            for row in subset.itertuples(index=False):
                membership = [name for name,pool in pools.items() if str(row.graph_name) in pool]
                if len(membership)!=1: raise ValueError(f"Unknown/multiple warm split: {row.graph_name}")
                group = membership[0]; counts[group] += 1
                rows.append({"dataset": dataset, "seed": int(seed), "test_graph_id": row.test_graph_id,
                    "original_graph_index": row.original_graph_index, "graph_name": row.graph_name,
                    "recorded_warm_split": group, "warm_checkpoint": str(path.relative_to(ROOT)),
                    "warm_seed": warm_args.seed})
            summary.append({"dataset": dataset, "seed": int(seed), "test_graphs": len(subset),
                "warm_train_overlap": counts["train"], "warm_validation_overlap": counts["validation"],
                "warm_test_overlap": counts["test"], "warm_train_overlap_fraction": counts["train"]/len(subset),
                "warm_checkpoint": str(path.relative_to(ROOT)), "warm_seed": warm_args.seed})
            summary[-1]["warm_split_reconstruction_available"] = True
        del training, validation, test, pools
        gc.collect()
        print(f"[EXPOSURE] audited recorded warm start for {dataset}", flush=True)
    output = ROOT/args.output_dir; output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"per_graph_exposure.csv", index=False)
    frame = pd.DataFrame(summary)
    frame.to_csv(output/"split_exposure.csv", index=False)
    evidence = {"datasets": args.datasets, "recorded_warm_split_audited": True,
        "splits_with_unrecoverable_warm_metadata":int((~frame.warm_split_reconstruction_available).sum()),
        "new_split_seeds_do_not_imply_unseen_backbone_graphs": bool((frame.warm_train_overlap>0).any()),
        "limitations": ["Checkpoint split reconstruction is not a full audit of resumed pretraining ancestry",
                        "Recorded test membership does not erase earlier method-development exposure"],
        "interpretation": "Matched residual/selection comparisons conditional on a shared warm start; not a clean five-seed end-to-end generalization benchmark"}
    (output/"COMPLETE.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (output/"REPORT.md").write_text("# Warm-start exposure audit\n\n"+
        frame.to_string(index=False)+"\n\n"+evidence["interpretation"]+".\n"+
        "Strict independent generalization requires splitting before every learned warm-start stage or testing never-trained graph instances.\n", encoding="utf-8")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
