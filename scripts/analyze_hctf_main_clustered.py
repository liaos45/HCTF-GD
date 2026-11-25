"""Recompute original main CIs using stable graph identities across split seeds."""
from pathlib import Path
import pandas as pd
from analyze_hctf_adaptive_results import ROOT, CORE_METRICS, hierarchical_bootstrap

DATASETS=["enzymes","fb15k237","imdb_binary","north","proteins","rome","sbm","wn18rr"]
SEEDS=[17,42,73,101,137]


def main():
    output=ROOT/"results/hctf_main_clustered_analysis"
    output.mkdir(parents=True,exist_ok=True)
    intervals=[]
    for comparison,trial,reference in [("full_vs_backbone","full_nodrop","baseline"),
                                       ("node_vs_backbone","node_only_nodrop","baseline"),
                                       ("full_vs_node","full_nodrop","node_only_nodrop")]:
        pairs={}
        for dataset in DATASETS:
            for seed in SEEDS:
                directory=ROOT/"results/hctf_submission_main"
                control=pd.read_csv(directory/f"submission_{dataset}_{reference}_s{seed}_test.csv")
                full=pd.read_csv(directory/f"submission_{dataset}_{trial}_s{seed}_test.csv")
                pairs[(dataset,seed)]=control.merge(full,on=["batch_id","local_graph_id"],
                                                  suffixes=("_node","_full"),validate="one_to_one")
        interval=hierarchical_bootstrap(pairs,4000,20260916,CORE_METRICS)
        interval["comparison"]=comparison
        intervals.append(interval)
        print(f"[CLUSTERED CI] {comparison}: {len(pairs)} dataset/seed pairs",flush=True)
    frame=pd.concat(intervals,ignore_index=True)
    frame.to_csv(output/"crossed_bootstrap_ci.csv",index=False)
    frame[frame.metric=="graph_stress"].to_csv(output/"stress_effects.csv",index=False)
    (output/"REPORT.md").write_text("# 原主实验的图身份聚类置信区间\n\n"+
        "使用原始图索引连接跨种子的重叠测试子集。split-local 序号不能作为跨种子图身份。原均值不变，新增区间不得与旧区间混用。ALL 等权平均固定的八个数据集。\n\n```text\n"+
        frame[frame.metric=="graph_stress"].to_string(index=False)+"\n```\n",encoding="utf-8")
    return 0


if __name__=="__main__": raise SystemExit(main())
