"""Saved-record statistics only. No models, graph loaders, training or prediction."""
from audit_common import *

DOMAINS=['enzymes','fb15k237','imdb_binary','north','proteins','rome','sbm','wn18rr'];SEEDS=[17,42,73,101,137]
INPUTS={};META=None

def read(path):
    path=Path(path);INPUTS[rel(path)]=sha(path);return pd.read_csv(path)

def original_analysis():
    p=ROOT/'scripts/analyze_hctf_adaptive_results.py';source=text(p);tree=ast.parse(source)
    selected=[]
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            names=[x.id for x in getattr(node,'targets',[]) if isinstance(x,ast.Name)]
            if set(names)&{'CORE_METRICS','HOLDOUT_METRICS','METRICS','LOWER_IS_BETTER','FLOORS'}:selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in ['signed_gain','hierarchical_bootstrap']:selected.append(node)
    namespace={'np':np,'pd':pd,'ROOT':ROOT};exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),namespace)
    INPUTS[rel(p)]=sha(p);return namespace

def join(a,b,keys,identity=None):
    p=a.merge(b,on=keys,suffixes=('_node','_full'),validate='one_to_one');assert len(p)==len(a)==len(b)
    for k in ['num_nodes','num_edges']:
        if k+'_node' in p:assert np.array_equal(p[k+'_node'],p[k+'_full'])
    if identity:p['_graph_cluster_id']=p[identity].astype(str)
    return p

def identify(pairs):
    global META
    if META is None:META=read(ROOT/'results/hctf_graph_metadata.csv')
    answer={}
    for (ds,seed),p in pairs.items():
        p=p.sort_values(['batch_id','local_graph_id']).copy()
        if '_graph_cluster_id' not in p:
            m=META[(META.dataset==ds)&(META.seed==seed)].sort_values('test_graph_id');assert len(p)==len(m)
            for k in ['num_nodes','num_edges']:
                for suffix in ['node','full']:assert np.array_equal(p[k+'_'+suffix].to_numpy(),m[k].to_numpy())
            p['_graph_cluster_id']=m.original_graph_index.to_numpy()
        assert not p._graph_cluster_id.duplicated().any();answer[ds,seed]=p
    return answer

def crossed_global_seed(pairs,repeats,seed,metric,analysis):
    """Sensitivity: globally coupled training seed, graph-ID weights within domain."""
    pairs=identify(pairs);rng=np.random.default_rng(seed);datasets=sorted({d for d,s in pairs});seeds=sorted({s for d,s in pairs});arrays={};observed=[]
    for ds in datasets:
        ids=sorted({v for (d,s),p in pairs.items() if d==ds for v in p._graph_cluster_id});mapping={v:i for i,v in enumerate(ids)};fold={}
        for s in seeds:
            p=pairs.get((ds,s))
            if p is not None:fold[s]=(np.array([mapping[v] for v in p._graph_cluster_id]),p[metric+'_node'].to_numpy(float),p[metric+'_full'].to_numpy(float))
        arrays[ds]=(len(ids),fold);observed.append(np.mean([analysis['signed_gain']([a.mean()],[b.mean()],metric)[0] for idx,a,b in fold.values()]))
    samples=[]
    for _ in range(repeats):
        chosen=rng.choice(seeds,len(seeds),replace=True);domain_values=[]
        for ds,(n,fold) in arrays.items():
            weights=rng.multinomial(n,np.full(n,1/n));values=[]
            for s in chosen:
                if s not in fold:continue
                ix,a,b=fold[s];w=weights[ix]
                if w.sum()==0:continue
                values.append(analysis['signed_gain']([np.average(a,weights=w)],[np.average(b,weights=w)],metric)[0])
            domain_values.append(float(np.mean(values)) if values else np.nan)
        samples.append(float(np.mean(domain_values)))
    good=np.array(samples);good=good[np.isfinite(good)]
    return dict(mean_gain_pct=float(np.mean(observed)),ci_low_pct=float(np.quantile(good,.025)),ci_high_pct=float(np.quantile(good,.975)),finite_replicates=len(good),repeats=repeats,random_seed=seed,unit='global training seed + domain-stratified original graph multinomial',datasets=len(datasets),unique_domain_graphs=sum(n for n,f in arrays.values()))

def main():
    event('statistics','Start saved-record reproduction; published archived evaluations only; sealed external holdouts excluded',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/02_BOOTSTRAP_AUDIT.py')
    analysis=original_analysis();allstats=[];details=[];dependencies=[];comparisons=[]
    if (HERE/'02_CLAIM_LEVEL_STATISTICS.csv').exists():
        allstats=pd.read_csv(HERE/'02_CLAIM_LEVEL_STATISTICS.csv').to_dict('records')
        details=pd.read_csv(HERE/'evidence/recomputed_original_protocol.csv').to_dict('records')
        dependencies=pd.read_csv(HERE/'evidence/paired_run_statistics.csv').to_dict('records')
    frame={}
    for ds in DOMAINS:
        for seed in SEEDS:
            for variant in ['baseline','full_nodrop','node_only_nodrop']:
                path=ROOT/f'results/hctf_submission_main/submission_{ds}_{variant}_s{seed}_test.csv';frame[ds,seed,variant]=read(path)
    csvout('evidence/main_record_counts.csv',[dict(dataset=d,seed=s,variant=v,records=len(x),duplicates=int(x.duplicated(['batch_id','local_graph_id']).sum())) for (d,s,v),x in frame.items()])
    assert len(frame)==120 and sum(len(v) for v in frame.values())==29175
    archive=read(ROOT/'results/hctf_main_clustered_analysis/crossed_bootstrap_ci.csv')
    for reference,trial,label in [('node_only_nodrop','full_nodrop','full_vs_node'),('baseline','full_nodrop','full_vs_backbone'),('baseline','node_only_nodrop','node_vs_backbone')]:
        pairs={(d,s):join(frame[d,s,reference],frame[d,s,trial],['batch_id','local_graph_id']) for d in DOMAINS for s in SEEDS};comparisons.append(('legacy_main',label,pairs,archive,20260916,['graph_stress']))
    for family,base,archivepath,runprefix in [('strict_original','hctf_strict_size_ood','hctf_strict_size_ood_analysis/crossed_bootstrap_ci.csv','strictsize'),('adaptive_random','hctf_adaptive_final','hctf_adaptive_final_analysis/hierarchical_bootstrap_ci.csv','adaptivefinal')]:
        refs=[('adaptive_supervised_node_only','adaptive_supervised_full','full_vs_node'),('baseline','adaptive_supervised_full','full_vs_backbone'),('baseline','adaptive_supervised_node_only','node_vs_backbone')] if family=='strict_original' else [('adaptive_supervised_node_only','adaptive_supervised_full','full_vs_node')]
        ar=read(ROOT/'results'/archivepath)
        for ref,trial,label in refs:
            pairs={}
            for d in ['enzymes','fb15k237','sbm']:
                for s in [17,42,73]:
                    folder=ROOT/'results'/base;a=read(folder/f'{runprefix}_{d}_{ref}_s{s}_test.csv');b=read(folder/f'{runprefix}_{d}_{trial}_s{s}_test.csv');p=join(a,b,['batch_id','local_graph_id'])
                    if family=='strict_original':
                        p=p.sort_values(['batch_id','local_graph_id']);m=read(folder/f'graph_metadata_{d}_s{s}.csv').sort_values('test_graph_id');assert len(p)==len(m)
                        for key in ['num_nodes','num_edges']:
                            for suffix in ['node','full']:assert np.array_equal(p[key+'_'+suffix],m[key])
                        p['_graph_cluster_id']=m.graph_name.to_numpy()
                    pairs[d,s]=p
            comparisons.append((family,label,pairs,ar,20260916,['graph_stress']))
    for family,folder,archivepath,references,rngseed in [('strict_Q1','hctf_q1_controls','hctf_q1_controls_analysis_strict/crossed_bootstrap_ci.csv',['node'],20260916),('mechanism','hctf_mechanistic_counterfactuals','hctf_mechanistic_counterfactuals_analysis/crossed_bootstrap_ci.csv',['same_weight_gate_zero','same_weight_gate_one','top_down_only','bottom_up_only'],20261009)]:
        fs={}
        for d in ['enzymes','fb15k237','sbm']:
            for s in [17,42,73]:
                f=ROOT/f'results/{folder}/strict_{d}_s{s}';mark=json.loads(text(f/'COMPLETE.json'));INPUTS[rel(f/'COMPLETE.json')]=sha(f/'COMPLETE.json');assert not mark['partial'] and mark['all_sources_complete'];fs[d,s]=read(f/'observations.csv')
        ar=read(ROOT/'results'/archivepath)
        for ref in references:
            pairs={k:join(v[(v.source==ref)&(v.stage=='raw')],v[(v.source=='full')&(v.stage=='raw')],['graph_name','batch_id','local_graph_id'],'graph_name') for k,v in fs.items()};comparisons.append((family,'full_raw_vs_'+ref+'_raw',pairs,ar,rngseed,['graph_stress']))
    # Seven selection-excluded endpoints. Re-evaluate statistics from archived metrics only.
    safe={}
    for d in DOMAINS:
        for s in SEEDS:
            for v in ['full_nodrop','node_only_nodrop']:safe[d,s,v]=read(ROOT/f'results/hctf_safe_blend_paired_holdout/{d}_{v}_s{s}/{d}_eval.csv')
    ar=read(ROOT/'figures/hctf_visualizations_20261009/safe_holdout/crossed_bootstrap_ci.csv')
    for label,reference in [('full_nodrop','baseline'),('safe_full_vs_safe_node','safe_node')]:
        pairs={}
        for d in DOMAINS:
            for s in SEEDS:
                f=safe[d,s,'full_nodrop'];a=f[f.method=='Current-Model-Baseline'] if reference=='baseline' else safe[d,s,'node_only_nodrop'][lambda x:x.method=='HCTF-SafeBlend'];b=f[f.method=='HCTF-SafeBlend'];p=join(a,b,['batch_id','graph_id']).rename(columns={'graph_id':'local_graph_id'});pairs[d,s]=p
        comparisons.append(('10plus7_evaluation_only',label,pairs,ar,20260916,analysis['HOLDOUT_METRICS']))
    for family,label,pairs,ar,seed,metrics in comparisons:
        done=[q for q in allstats if q['family']==family and q['comparison']==label]
        if len(done)==(len({d for d,s in pairs})+1)*len(metrics):
            print('RESUME completed comparison',family,label,flush=True);continue
        pairs=identify(pairs)
        for p in pairs.values():
            if 'graph_stress_node' not in p:
                for suffix in ['node','full']:
                    p['graph_stress_'+suffix]=p['batch_stress_'+suffix]/(p['num_nodes_'+suffix]*(p['num_nodes_'+suffix]-1))
        for (d,s),p in pairs.items():dependencies.append(dict(family=family,comparison=label,dataset=d,seed=s,records=len(p),original_ids=p._graph_cluster_id.nunique(),reference_mean=p.graph_stress_node.mean(),trial_mean=p.graph_stress_full.mean(),mean_graphwise_Stress_relative_gain=float(analysis['signed_gain'](p.graph_stress_node,p.graph_stress_full,'graph_stress').mean()),ratio_of_run_means_Stress_gain=float(analysis['signed_gain']([p.graph_stress_node.mean()],[p.graph_stress_full.mean()],'graph_stress')[0])))
        calculated=analysis['hierarchical_bootstrap'](pairs,4000,seed,metrics=metrics)
        target=ar[ar.comparison==label] if 'comparison' in ar else ar
        for r in calculated.to_dict('records'):
            t=target[(target.dataset==r['dataset'])&(target.metric==r['metric'])]
            if len(t)!=1:continue
            saved=t.iloc[0];same=max(abs(float(saved[f])-r[f]) for f in ['mean_gain_pct','ci_low_pct','ci_high_pct'])<1e-9
            row=dict(family=family,comparison=label,dataset=r['dataset'],metric=r['metric'],archived_estimate=float(saved.mean_gain_pct),archived_low=float(saved.ci_low_pct),archived_high=float(saved.ci_high_pct),recomputed_original_protocol_estimate=r['mean_gain_pct'],recomputed_original_protocol_low=r['ci_low_pct'],recomputed_original_protocol_high=r['ci_high_pct'],archive_exact_match=same,source_analysis_sha256=sha(ROOT/'scripts/analyze_hctf_adaptive_results.py'),bootstrap_repeats=4000,bootstrap_seed=seed,archived_method_status='CURRENT_SOURCE_REPRODUCED' if same else 'CURRENT_SOURCE_DIFFERS_FROM_ARCHIVE',numeric_reproduction_status='VERIFIED' if same else 'ARCHIVED ONLY');allstats.append(row)
        # Keep sensitivity separate; never overwrite archived values.
        if metrics==['graph_stress']:
            sensitivity=crossed_global_seed(pairs,4000,20261028,'graph_stress',analysis)
            for r in allstats:
                if r['family']==family and r['comparison']==label and r['dataset']=='ALL':r.update({f'graph_aware_sensitivity_{k}':v for k,v in sensitivity.items()})
        csvout('02_CLAIM_LEVEL_STATISTICS.csv',allstats);csvout('evidence/paired_run_statistics.csv',dependencies)
        for r in calculated.to_dict('records'):details.append(dict(family=family,comparison=label,**r))
        csvout('evidence/recomputed_original_protocol.csv',details);save('evidence/statistical_input_hashes.json',INPUTS);print('RECOMPUTED',family,label,'matched',sum(q['archive_exact_match'] for q in allstats if q['family']==family and q['comparison']==label),flush=True)
    save('evidence/STATISTICS_COMPLETE.json',dict(comparisons=len(comparisons),rows=len(allstats),all_archives_reproduced=all(q['archive_exact_match'] for q in allstats),repeats=4000,Test_forward=False));event('statistics',f'Completed {len(comparisons)} archived-data comparisons, {len(allstats)} comparison/domain/metric rows; no historical files overwritten')

if __name__=='__main__':main()
