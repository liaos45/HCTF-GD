"""Only accept audited split-before-learning Size-OOD evaluations."""
import argparse
import json
import numpy as np
import pandas as pd
from evaluate_hctf_safe_blend import ROOT
from analyze_hctf_adaptive_results import hierarchical_bootstrap, METRICS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-dir",default="results/hctf_strict_size_ood")
    parser.add_argument("--output-dir",default="results/hctf_strict_size_ood_analysis")
    parser.add_argument("--bootstrap-repeats",type=int,default=4000)
    args=parser.parse_args()
    directory,output=ROOT/args.input_dir,ROOT/args.output_dir
    intervals=[]
    for comparison,reference,trial in [("full_vs_node","adaptive_supervised_node_only","adaptive_supervised_full"),
        ("full_vs_backbone","baseline","adaptive_supervised_full"),("node_vs_backbone","baseline","adaptive_supervised_node_only")]:
        pairs={}
        for dataset in ["enzymes","fb15k237","sbm"]:
            for seed in [17,42,73]:
                audit=json.loads((directory/f"COMPLETE_{dataset}_s{seed}.json").read_text())
                if audit["partial"] or not audit["no_legacy_weight_initialization"] or not audit["frozen_common_backbone_state_verified"]:
                    raise RuntimeError("Partial or unverified strict Size-OOD output")
                control=pd.read_csv(directory/f"strictsize_{dataset}_{reference}_s{seed}_test.csv")
                full=pd.read_csv(directory/f"strictsize_{dataset}_{trial}_s{seed}_test.csv")
                frame=control.merge(full,on=["batch_id","local_graph_id"],suffixes=("_node","_full"),validate="one_to_one")
                frame=frame.sort_values(["batch_id","local_graph_id"])
                metadata=pd.read_csv(directory/f"graph_metadata_{dataset}_s{seed}.csv").sort_values("test_graph_id")
                if len(frame)!=len(control) or len(frame)!=len(full) or len(frame)!=len(metadata): raise RuntimeError("Missing strict paired graph")
                for suffix in ["node","full"]:
                    if not np.array_equal(frame[[f"num_nodes_{suffix}",f"num_edges_{suffix}"]],metadata[["num_nodes","num_edges"]]):
                        raise RuntimeError("Strict graph order/structure mismatch")
                frame["_graph_cluster_id"]=metadata.graph_name.to_numpy()
                pairs[(dataset,seed)]=frame
        ci=hierarchical_bootstrap(pairs,args.bootstrap_repeats,20260916,metrics=METRICS)
        ci["comparison"]=comparison; intervals.append(ci)
        print(f"[STRICT CI] {comparison}",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.concat(intervals,ignore_index=True)
    frame.to_csv(output/"crossed_bootstrap_ci.csv",index=False)
    stress=frame[frame.metric=="graph_stress"]
    stress.to_csv(output/"stress_effects.csv",index=False)
    (output/"COMPLETE.json").write_text(json.dumps({"training_runs":27,"paired_full_node_jobs":9,
        "split_before_all_learned_initialization":True,"legacy_weights_never_loaded":True,
        "all_negative_results_retained":True,"development_collection_not_blinded":True,
        "warning":"60-epoch budgeted scratch backbone and 20-epoch adapters, not established equal full convergence"},indent=2),encoding="utf-8")
    (output/"REPORT.md").write_text("# Split-before-learning Size-OOD\n\n"+stress.to_string(index=False)+
        "\n\nAll learned stages use the same small-graph split. This fixes recorded pretrained graph overlap, but not prior development exposure or limited-convergence budgets.\n",encoding="utf-8")
    print(stress.to_string(index=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
