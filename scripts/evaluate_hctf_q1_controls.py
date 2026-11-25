"""Shared-input source controls, raw strong baselines, independent task proxies.

Full test pools are evaluated. Smoke outputs have explicit partial markers.
Concurrent timings are descriptive, never advertised as isolated benchmarks.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import evaluate_hctf_safe_blend as evaluator
from hctf_fast_audit import install_fast_audit
from hctf_q1_controls import SIMPLE_SOURCES, TASK_METRICS, ALPHAS, graph_from_edges, task_queries, task_proxy_metrics, simple_candidate, choose_safe

install_fast_audit(evaluator)
ROOT,train = evaluator.ROOT,evaluator.train
SOURCES = ('full','node','same_weight_gate_zero')+SIMPLE_SOURCES


def checkpoint_paths(dataset,seed,mode):
    directory,prefix = ('hctf_strict_size_ood','strictsize') if mode=='strict' else ('hctf_clean_id','cleanid')
    folder = ROOT/'checkpoints'/directory
    return {source:folder/f'{prefix}_{dataset}_{variant}_s{seed}_best.pt'
            for source,variant in [('baseline','baseline'),('full','adaptive_supervised_full'),('node','adaptive_supervised_node_only')]}


def stamp_contract(args,paths):
    return {'dataset':args.dataset,'seed':args.seed,'mode':args.mode,'sources':list(SOURCES),
        'alphas':ALPHAS,'selection_metrics':evaluator.METRICS,'post_decision_task_metrics':TASK_METRICS,
        'max_graphs':args.max_graphs,'queries':128,'single_baseline_inference':True,
        'timing_isolated':False,'partial':args.max_graphs>0,
        'checkpoint_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
        'source_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                         for name in ['hctf_q1_controls.py','evaluate_hctf_q1_controls.py','hctf_fast_audit.py']}}


@torch.no_grad()
def evaluate(args):
    paths = checkpoint_paths(args.dataset,args.seed,args.mode)
    audit_dir = ROOT/'results'/('hctf_strict_size_ood' if args.mode=='strict' else 'hctf_clean_id')
    audit = json.loads((audit_dir/f'COMPLETE_{args.dataset}_s{args.seed}.json').read_text())
    if audit['partial'] or not audit['frozen_common_backbone_state_verified']:
        raise RuntimeError('Require fully audited clean checkpoint group')
    output = ROOT/args.output_dir/f'{args.mode}_{args.dataset}_s{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    contract = stamp_contract(args,paths)
    protocol_path = output/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=json.loads(json.dumps(contract)):
        raise RuntimeError('Changed evaluator protocol/checkpoints, use a new output directory')
    protocol_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    if args.skip_completed and (output/'COMPLETE.json').exists(): return 0
    device = train.get_device()
    models = {name:evaluator.load_model(path,device) for name,path in paths.items()}
    for model_args,_ in models.values(): model_args._hctf_teacher_enabled=False
    train.set_random_seed(args.seed)
    data_list = train.load_graph_dataset(models['baseline'][0])[2]
    metadata = pd.read_csv(audit_dir/f'graph_metadata_{args.dataset}_s{args.seed}.csv').sort_values('test_graph_id')
    if [str(data.name) for data in data_list]!=metadata.graph_name.astype(str).tolist():
        raise RuntimeError('Checkpoint evaluation graph identity/order mismatch')
    if args.max_graphs>0: data_list=data_list[:args.max_graphs]
    loader = train.make_loader(data_list,int(models['baseline'][0].batch_size),shuffle=False)
    rows=[]; graph_id=0; visual_cases=[]
    for batch_id,batch in enumerate(loader):
        batch=batch.to(device)
        positions={}; forward_seconds={}
        for source,(model_args,model) in models.items():
            if torch.cuda.is_available(): torch.cuda.synchronize()
            started=time.perf_counter()
            _,positions[source],_,_=train.forward_model(batch,model,None,evaluator.Stress(),model_args)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            forward_seconds[source]=time.perf_counter()-started
        full_args,full=models['full']
        full.set_hctf_structure_gate_override(0.)
        started=time.perf_counter()
        try:
            _,positions['same_weight_gate_zero'],_,_=train.forward_model(batch,full,None,evaluator.Stress(),full_args)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            forward_seconds['same_weight_gate_zero']=time.perf_counter()-started
        finally: full.set_hctf_structure_gate_override(None)
        records=train.mtpf_build_local_graph_records(batch)
        for record in records:
            nodes=record['global_nodes_device']
            graph=graph_from_edges(record['num_nodes'],record['edge_index'])
            name=str(data_list[graph_id].name)
            stable_seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%2147483647
            # Fixed reference is planted for SBM, greedy detected for real domains.
            if args.dataset=='sbm':
                labels=train._community_labels_from_graph(data_list[graph_id],'ground_truth',stable_seed)
                reference='planted_sbm'
            else:
                labels=evaluator.detect_communities(record['edge_index'],record['num_nodes'])
                reference='fixed_greedy_proxy'
            queries=task_queries(graph,stable_seed)
            baseline,baseline_metrics=evaluator.measure_layout(positions['baseline'][nodes],record,labels,include_holdout=False)
            baseline=baseline.cpu()
            baseline_metrics['graph_stress']=baseline_metrics.pop('batch_stress')/(record['num_nodes']*(record['num_nodes']-1))
            # Selector retains its original summed stress convention.
            audit_baseline={**baseline_metrics,'batch_stress':baseline_metrics['graph_stress']*record['num_nodes']*(record['num_nodes']-1)}
            baseline_export={**baseline_metrics,**evaluator.independent_visual_metrics(baseline,record['edge_index']),
                             **task_proxy_metrics(baseline.numpy(),graph,labels,queries,stable_seed)}
            common={'dataset':args.dataset,'seed':args.seed,'mode':args.mode,'graph_name':name,
                    'batch_id':batch_id,'local_graph_id':graph_id,'reference_partition':reference,
                    'routing_queries':len(queries),'timing_isolated':False}
            rows.append({**common,**baseline_export,'source':'baseline','stage':'baseline','alpha':0.,
                         'generation_seconds':forward_seconds['baseline']/len(records),'audit_seconds':0.})
            case={'graph_name':name,'edges':list(graph.edges),'labels':np.asarray(labels).tolist(),
                  'queries':queries[:8],'layouts':{'baseline':baseline.tolist()},'selected_alphas':{}}
            for source in SOURCES:
                started=time.perf_counter()
                if source in positions:
                    proposal=positions[source][nodes].detach().cpu()
                    generation_seconds=forward_seconds[source]/len(records)
                else:
                    proposal=torch.as_tensor(simple_candidate(source,baseline.numpy(),graph,labels,stable_seed,train,record),dtype=baseline.dtype)
                    generation_seconds=time.perf_counter()-started
                started=time.perf_counter()
                raw,raw_metrics=evaluator.measure_layout(proposal,record,labels,include_holdout=False)
                chosen,chosen_metrics,alpha=choose_safe(baseline,audit_baseline,raw,record,labels,evaluator)
                case['selected_alphas'][source]=alpha
                audit_seconds=time.perf_counter()-started
                for stage,coords,metrics in [('raw',raw.cpu(),raw_metrics),('safe',chosen.cpu(),chosen_metrics)]:
                    export=dict(metrics)
                    export['graph_stress']=export.pop('batch_stress')/(record['num_nodes']*(record['num_nodes']-1))
                    export.update(evaluator.independent_visual_metrics(coords,record['edge_index']))
                    # Tasks are evaluated strictly after the choice and timer.
                    export.update(task_proxy_metrics(coords.numpy(),graph,labels,queries,stable_seed))
                    rows.append({**common,**export,'source':source,'stage':stage,'alpha':alpha if stage=='safe' else 1.,
                                 'generation_seconds':generation_seconds,'audit_seconds':audit_seconds})
                    case['layouts'][f'{source}_{stage}']=coords.tolist()
            # Fixed selection strata, never conditioned on measured full-vs-node gains.
            size_group=min(record['num_nodes']//32,8)
            selection_group='accepted' if case['selected_alphas']['full']>0 else 'fallback'
            if not any(value['size_group']==size_group and value['selection_group']==selection_group for value in visual_cases):
                case.update(size_group=size_group,selection_group=selection_group)
                visual_cases.append(case)
            graph_id+=1
            if graph_id%10==0: print(f'[CONTROLS] {args.mode} {args.dataset} s{args.seed} {graph_id}/{len(data_list)}',flush=True)
    pd.DataFrame(rows).to_csv(output/'observations.csv',index=False)
    (output/'visual_cases.json').write_text(json.dumps(visual_cases),encoding='utf-8')
    (output/'COMPLETE.json').write_text(json.dumps({**contract,'graphs':graph_id,'rows':len(rows),
        'all_sources_complete':graph_id==len(data_list),'reference_partition':reference},indent=2),encoding='utf-8')
    print(f'[CONTROLS COMPLETE] {output} {graph_id} graphs',flush=True)
    del models,data_list,batch,positions
    gc.collect()
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--mode',choices=['strict','clean-id'],default='strict')
    parser.add_argument('--max-graphs',type=int,default=-1)
    parser.add_argument('--output-dir',default='results/hctf_q1_controls')
    parser.add_argument('--skip-completed',action='store_true')
    return evaluate(parser.parse_args())


if __name__=='__main__': raise SystemExit(main())
