"""Reproduce actual nested objective CI and separate graph-ID sensitivity."""
from audit_common import *
from types import SimpleNamespace
import importlib.util

def main():
    spec=importlib.util.spec_from_file_location('submission_stat_audit',HERE/'02_BOOTSTRAP_AUDIT.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);new=audit.original_analysis()
    p=ROOT/'scripts/analyze_hctf_paper_experiments.py';tree=ast.parse(text(p));nodes=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['METRICS','LOWER_IS_BETTER','SCORE_FLOORS','STRUCTURAL_CONTRASTS'] for t in n.targets):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in ['signed_difference','floor_normalized_gain','hierarchical_bootstrap_contrasts']:nodes.append(n)
    ns={'np':np,'pd':pd};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);analysis=SimpleNamespace(**ns)
    f=ROOT/'scripts/analyze_hctf_objective_ablations.py';fun=next(n for n in ast.parse(text(f)).body if isinstance(n,ast.FunctionDef) and n.name=='equal_domain_bootstrap');other={'analysis':analysis,'np':np,'pd':pd};exec(compile(ast.Module(body=[fun],type_ignores=[]),str(f),'exec'),other)
    allpairs={};cross={};inputs={};identity=[]
    for removal in ['no_caid_target_nodrop','no_macpsg_nodrop','no_caid_macpsg_nodrop']:
        cross[removal]={}
        for d in ['enzymes','fb15k237','sbm']:
            for seed in [17,42,73]:
                pa=ROOT/f'results/hctf_objective_ablations/objablate_{d}_{removal}_s{seed}_test.csv';pb=ROOT/f'results/hctf_submission_extensions/ablation/subablate_{d}_full_nodrop_s{seed}_test.csv';a=pd.read_csv(pa);b=pd.read_csv(pb);inputs[rel(pa)]=sha(pa);inputs[rel(pb)]=sha(pb)
                joined=audit.join(a,b,['batch_id','local_graph_id']);pair=audit.identify({(d,seed):joined})[d,seed];cross[removal][d,seed]=pair;renamed=pair.rename(columns={c:c.replace('_node','_reference').replace('_full','_trial') for c in pair if c.endswith('_node') or c.endswith('_full')});allpairs[d,removal,'full_nodrop',seed]=renamed;identity.append(dict(removal=removal,dataset=d,seed=seed,graphs=len(pair),original_ids=pair._graph_cluster_id.nunique(),identity_basis='archived metadata order+count+n+m; no historical per-edge fingerprint'))
    old=ns['hierarchical_bootstrap_contrasts'](allpairs,4000,20261009);eq=other['equal_domain_bootstrap'](allpairs,4000,20261010);calc=pd.concat([old,eq],ignore_index=True);archive=pd.read_csv(ROOT/'results/hctf_objective_ablations_analysis/objective_contrast_bootstrap_ci.csv');merge=calc.merge(archive,on=['dataset','reference_variant','trial_variant','metric'],suffixes=('_new','_archive'),validate='one_to_one')
    for field in ['mean_gain_pct','hierarchical_bootstrap_ci_low_pct','hierarchical_bootstrap_ci_high_pct']:assert np.abs(merge[field+'_new']-merge[field+'_archive']).max()<1e-9
    csvout('evidence/objective_original_protocol_reproduction.csv',merge);tables=[]
    for removal,pairs in cross.items():
        q=new['hierarchical_bootstrap'](pairs,4000,20261023,metrics=['graph_stress','community_compactness','community_sep_comp_ratio']);q['reference_variant']=removal;tables.append(q);print('OBJECTIVE crossed',removal,flush=True)
    sens=pd.concat(tables);sens=sens.merge(archive[['dataset','reference_variant','metric','mean_gain_pct','hierarchical_bootstrap_ci_low_pct','hierarchical_bootstrap_ci_high_pct']],on=['dataset','reference_variant','metric'],suffixes=('_sensitivity','_archived'),validate='one_to_one');assert (sens.mean_gain_pct_sensitivity-sens.mean_gain_pct_archived).abs().max()<1e-9
    csvout('evidence/objective_graph_aware_sensitivity.csv',sens);csvout('evidence/objective_pair_identity_checks.csv',identity);save('evidence/objective_source_hashes.json',{rel(p):sha(p),rel(f):sha(f),**inputs});save('evidence/OBJECTIVE_STATISTICS_COMPLETE.json',dict(archive_original_rows=len(merge),all_original_values_match=True,sensitivity_rows=len(sens),point_values_unchanged=True,old_intervals_not_overwritten=True));event('objective_statistics','Actual nested120rows reproduced; separate36row graph-ID sensitivity computed, historical files unchanged',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_objective_statistics.py')

if __name__=='__main__':main()
