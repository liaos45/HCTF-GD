"""Commit-checked paired analysis. No incomplete sweep becomes formal evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_hctf_adaptive_results import hierarchical_bootstrap, METRICS
from hctf_q1_controls import TASK_METRICS, SIMPLE_SOURCES
from evaluate_hctf_q1_controls import SOURCES

ROOT=Path(__file__).resolve().parents[1]


def validate_source_rows(frame,sources):
    expected={('baseline','baseline')}|{(source,stage) for source in sources for stage in ['raw','safe']}
    if frame.duplicated(['graph_name','source','stage']).any(): raise RuntimeError('Duplicate source graph rows')
    for _,group in frame.groupby('graph_name'):
        if set(zip(group.source,group.stage))!=expected: raise RuntimeError('Missing or unexpected source/stage graph rows')


def load_committed(root,mode,datasets,seeds,allow_partial):
    frames={}; missing=[]
    for dataset in datasets:
        for seed in seeds:
            folder=root/f'{mode}_{dataset}_s{seed}'
            if not (folder/'COMPLETE.json').exists():
                missing.append(f'{dataset}/s{seed}'); continue
            marker=json.loads((folder/'COMPLETE.json').read_text())
            if marker['partial'] or not marker['all_sources_complete']:
                raise RuntimeError(f'Partial/incomplete observations cannot enter analysis {folder}')
            frame=pd.read_csv(folder/'observations.csv')
            if len(frame)!=marker['graphs']*(1+2*len(SOURCES)):
                raise RuntimeError(f'Missing source rows {folder}')
            if frame.duplicated(['graph_name','source','stage']).any():
                raise RuntimeError(f'Duplicate source graph {folder}')
            validate_source_rows(frame,SOURCES)
            if not np.isfinite(frame[METRICS+list(TASK_METRICS)].to_numpy()).all():
                raise RuntimeError(f'Non-finite observations {folder}')
            frames[(dataset,seed)]=frame
    if missing and not allow_partial:
        raise RuntimeError(f'Formal analysis requires all groups. Missing {missing}')
    if not frames: raise RuntimeError('No fully committed control groups')
    return frames,missing


def paired_frames(frames,reference,trial):
    pairs={}
    for key,frame in frames.items():
        control=frame[(frame.source==reference[0])&(frame.stage==reference[1])]
        candidate=frame[(frame.source==trial[0])&(frame.stage==trial[1])]
        pair=control.merge(candidate,on=['graph_name','batch_id','local_graph_id'],suffixes=('_node','_full'),validate='one_to_one')
        if len(pair)!=len(control) or len(pair)!=len(candidate):
            raise RuntimeError('Missing paired graphs')
        for name in ['num_nodes','num_edges','routing_queries']:
            if not np.array_equal(pair[f'{name}_node'],pair[f'{name}_full']):
                raise RuntimeError(f'Graph/task pairing differs {name}')
        pair['_graph_cluster_id']=pair.graph_name.astype(str)
        pairs[key]=pair
    return pairs


def task_intervals(pairs,repeats,seed,metrics=TASK_METRICS):
    rng=np.random.default_rng(seed); rows=[]
    for metric in metrics:
        samples_by_domain=[]; observed_by_domain=[]
        for dataset in sorted({dataset for dataset,_ in pairs}):
            groups=[frame for (domain,_),frame in sorted(pairs.items()) if domain==dataset]
            identities=sorted({name for frame in groups for name in frame.graph_name})
            mapping={name:i for i,name in enumerate(identities)}
            indices=[np.asarray([mapping[name] for name in frame.graph_name]) for frame in groups]
            deltas=[frame[f'{metric}_full'].to_numpy()-frame[f'{metric}_node'].to_numpy() for frame in groups]
            samples=[]
            for _ in range(repeats):
                selected=rng.integers(len(groups),size=len(groups))
                counts=rng.multinomial(len(identities),np.full(len(identities),1./len(identities)))
                while any(not counts[indices[index]].sum() for index in selected):
                    counts=rng.multinomial(len(identities),np.full(len(identities),1./len(identities)))
                samples.append(np.mean([np.average(deltas[index],weights=counts[indices[index]]) for index in selected]))
            samples=np.asarray(samples)
            observed=float(np.mean([value.mean() for value in deltas]))
            rows.append({'dataset':dataset,'metric':metric,'mean_absolute_gain':observed,
                         'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),
                         'seeds':len(groups),'graphs':len(identities)})
            samples_by_domain.append(samples); observed_by_domain.append(observed)
        samples=np.mean(samples_by_domain,axis=0)
        rows.append({'dataset':'ALL','metric':metric,'mean_absolute_gain':float(np.mean(observed_by_domain)),
                     'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),
                     'seeds':len({seed for _,seed in pairs}),'graphs':sum(row['graphs'] for row in rows if row['metric']==metric and row['dataset']!='ALL')})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',default='results/hctf_q1_controls')
    parser.add_argument('--output-dir',default='results/hctf_q1_controls_analysis')
    parser.add_argument('--mode',choices=['strict','clean-id'],default='strict')
    parser.add_argument('--datasets',nargs='+',default=['enzymes','fb15k237','sbm'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,42,73])
    parser.add_argument('--bootstrap-repeats',type=int,default=4000)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    frames,missing=load_committed(ROOT/args.input_dir,args.mode,args.datasets,args.seeds,args.allow_partial)
    output=ROOT/args.output_dir; output.mkdir(parents=True,exist_ok=True)
    observations=pd.concat(frames.values(),ignore_index=True)
    observations.groupby(['dataset','seed','source','stage'])[METRICS+list(TASK_METRICS)+['generation_seconds','audit_seconds']].mean().reset_index().to_csv(output/'absolute_seed_means.csv',index=False)
    observations[observations.stage=='safe'].assign(accepted=lambda x:x.alpha>0).groupby(['dataset','seed','source']).agg(graphs=('graph_name','size'),accepted=('accepted','sum'),acceptance_rate=('accepted','mean')).reset_index().to_csv(output/'acceptance.csv',index=False)
    comparisons=[]
    for stage in ['raw','safe']:
        for source in SOURCES:
            comparisons.append(((source,stage),('baseline','baseline'),f'{source}_{stage}_vs_baseline'))
        for source in ['node','same_weight_gate_zero']+list(SIMPLE_SOURCES):
            comparisons.append((('full',stage),(source,stage),f'full_{stage}_vs_{source}_{stage}'))
    relative=[]; absolute=[]
    for trial,reference,label in comparisons:
        pairs=paired_frames(frames,reference,trial)
        table=hierarchical_bootstrap(pairs,args.bootstrap_repeats,20260916,metrics=METRICS)
        table['comparison']=label; relative.append(table)
        task=task_intervals(pairs,args.bootstrap_repeats,20260916)
        task['comparison']=label; absolute.append(task)
        print(f'[CONTROL CI] {label}',flush=True)
    pd.concat(relative,ignore_index=True).to_csv(output/'crossed_bootstrap_ci.csv',index=False)
    pd.concat(absolute,ignore_index=True).to_csv(output/'task_absolute_bootstrap_ci.csv',index=False)
    expected_domains=['enzymes','fb15k237','sbm'] if args.mode=='strict' else list(__import__('run_hctf_all8_pilot').DATASETS)
    scope_complete=not missing and set(args.datasets)==set(expected_domains) and set(args.seeds)=={17,42,73}
    (output/'COMPLETE.json').write_text(json.dumps({'complete':scope_complete,'partial':not scope_complete,
        'complete_for_requested_scope':not missing,
        'mode':args.mode,'datasets':args.datasets,'seeds':args.seeds,'missing_groups':missing,
        'paired_groups':len(frames),'independent_human_study':False,'timing_isolated':False,
        'intervals':'Endpoint-wise crossed graph/seed bootstrap; not simultaneous confidence intervals',
        'task_units':'Absolute rate/ARI/NMI changes, not relative percentages'},indent=2),encoding='utf-8')
    (output/'REPORT.md').write_text('# 同选择器候选归因与独立任务代理\n\n'+
        f'已提交配对组 {len(frames)}。缺失组 {missing}。\n\n'+
        'raw 与 safe 分开。full 直接与 node、同权 gate-zero 和所有简单候选比较，不依据结果挑选对手。\n\n'+
        '任务量是算法路由和固定参考标签恢复，不是参与者表现。ARI/NMI 使用绝对变化，路由失败效率为零。\n\n'+
        '共用图和源种子交叉 bootstrap。并发计时不作为隔离速度结论。所有负结果保留。\n',encoding='utf-8')
    return 0


if __name__=='__main__': raise SystemExit(main())
