"""Audit saved decisions and execute original selector AST on controlled fixtures."""
from audit_common import *
import torch


def functions(path,names,assignments,namespace):
    tree=ast.parse(text(path));selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in node.targets):selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in names:selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace);return namespace

def main():
    path=ROOT/'scripts/evaluate_hctf_safe_blend.py';ns=functions(path,['objective_metric_weights','improvement','componentwise_violations','is_componentwise_safe'],['METRICS','HOLDOUT_METRICS','ALL_METRICS','LOWER_IS_BETTER','SCORE_FLOORS','ABS_TOLERANCE'],{'np':np})
    paired=ROOT/'scripts/evaluate_hctf_paired_holdout.py';tree=ast.parse(text(paired));evaluate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate');loop=next(n for n in ast.walk(evaluate) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='alpha')
    write('evidence/original_selection_loop.txt',lines(paired,loop.lineno,loop.end_lineno));fixture=[]
    baseline={m:1. for m in ns['METRICS']};baseline['edge_length_variance']=.01;baseline['edge_crossing']=10.;baseline['edge_crossing_norm']=.02
    def execute(name,trials,expected_alpha):
        env=dict(ns);basepos=np.array([[0.,0.],[1.,0.],[0.,1.]]);cursor=iter(trials)
        env.update(seed_pos=basepos,seed_metrics=baseline.copy(),best_pos=basepos,best_metrics=baseline.copy(),best_score=0.,best_alpha=0.,alphas=[t[0] for t in trials],weights=ns['objective_metric_weights'](''),record=None,labels=None,aligned=basepos+np.array([.1,.2]))
        env['measure_layout']=lambda position,record,labels,include_holdout=False:(position,next(cursor)[1])
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(paired),'exec'),env)
        verdict=env['best_alpha']==expected_alpha;fixture.append(dict(case=name,selected_alpha=env['best_alpha'],expected_alpha=expected_alpha,score=env['best_score'],fallback_object_identity=env['best_pos'] is basepos,exact_baseline_array_equal=np.array_equal(env['best_pos'],basepos),pass_check=verdict));assert verdict
    bad={**baseline,'edge_crossing':11.};negative={**baseline,'batch_stress':1-2e-10,'edge_length_variance':.01+5e-10};positive={**baseline,'batch_stress':.9};tiny={**baseline,'batch_stress':1-1e-12};tie={**baseline,'batch_stress':1-2e-10,'edge_length_variance':.01+2e-12}
    assert ns['is_componentwise_safe'](baseline,negative) and max(ns['improvement'](baseline,negative,m) for m in ns['METRICS'])>1e-10 and sum(ns['improvement'](baseline,negative,m) for m in ns['METRICS'])<0
    execute('no_feasible_trial',[(.25,bad),(.5,bad)],0.)
    execute('feasible_improvement_but_negative_total_score',[(.25,negative)],0.)
    execute('positive_trial_then_equal_score_larger_alpha',[(.25,positive),(.75,positive)],.75)
    execute('positive_trial_not_replaced_by_worse_score',[(.25,positive),(.75,{**baseline,'batch_stress':.95})],.25)
    execute('improvement_below_required_threshold',[(1.,tiny)],0.)
    execute('nonfinite_trial',[(1.,{**baseline,'batch_stress':float('nan')})],0.)
    execute('near_zero_incumbent_tie_can_select_larger_alpha',[(.5,tie)],.5)
    csvout('evidence/selector_boundary_fixtures.csv',fixture)
    config=[]
    for m in ns['METRICS']:config.append(dict(protocol='legacy paired 10+7',metric=m,orientation='lower' if m in ns['LOWER_IS_BETTER'] else 'higher',absolute_tolerance=ns['ABS_TOLERANCE'][m],relative_tolerance=0.,score_floor=ns['SCORE_FLOORS'][m],weight=1.,min_component_improvement=1e-10,score_tie_tolerance=1e-12,source=rel(path),source_sha256=sha(path),selection_loop_source=rel(paired),selection_loop_sha256=sha(paired),evaluated_config_hash_binding='historical PAIRED markers record metric lists, not source/checkpoint byte hashes'))
    csvout('04_SAFEBLEND_CONFIG_AUDIT.csv',config)
    counts=[];fall=[];hashes={}
    for ds in ['enzymes','fb15k237','imdb_binary','north','proteins','rome','sbm','wn18rr']:
        for seed in [17,42,73,101,137]:
            shared=[]
            for variant in ['full_nodrop','node_only_nodrop']:
                p=ROOT/f'results/hctf_safe_blend_paired_holdout/{ds}_{variant}_s{seed}';s=pd.read_csv(p/f'{ds}_selection.csv');f=pd.read_csv(p/f'{ds}_eval.csv');marker=ROOT/f'results/hctf_safe_blend_paired_holdout/PAIRED_{ds}_s{seed}.json';mark=json.loads(text(marker));assert mark['graphs']==len(s);assert np.array_equal(s.accepted,s.selected_alpha>0);assert set(s.selected_alpha).issubset({0.,.0625,.125,.25,.5,.75,1.});assert not s.duplicated(['batch_id','graph_id']).any()
                selected=f[f.method=='HCTF-SafeBlend'];base=f[f.method=='Current-Model-Baseline'];joined=base.merge(selected,on=['batch_id','graph_id'],suffixes=('_base','_selected'),validate='one_to_one');assert len(joined)==len(s)
                flags=joined.selected_alpha_selected==0
                finite_equal=True
                for m in ns['ALL_METRICS']:
                    a=joined.loc[flags,m+'_base'].to_numpy();b=joined.loc[flags,m+'_selected'].to_numpy();finite_equal=finite_equal and np.array_equal(a,b,equal_nan=True)
                counts.append(dict(dataset=ds,seed=seed,variant=variant,records=len(s),accepted=int(s.accepted.sum()),fallback=int((~s.accepted).sum()),rate_percent=100*s.accepted.mean(),zero_alpha_is_fallback=True,archived_fallback_17metrics_exact_equal=finite_equal,selection=rel(p/f'{ds}_selection.csv'),selection_sha256=sha(p/f'{ds}_selection.csv'),paired_marker_sha256=sha(marker),historical_checkpoint_hash_in_marker=bool(mark.get('checkpoint_sha256')),historical_source_hash_in_marker=bool(mark.get('source_sha256'))))
                shared.append(base.sort_values(['batch_id','graph_id']).reset_index(drop=True))
                for q in [p/f'{ds}_selection.csv',p/f'{ds}_eval.csv',marker]:hashes[rel(q)]=sha(q)
            assert all(np.array_equal(shared[0][m],shared[1][m],equal_nan=True) for m in ns['ALL_METRICS'])
    csvout('04_ACCEPTANCE_REPRODUCTION.csv',counts)
    # Audit archived All17 deployment portfolios. No candidate/model is re-executed.
    all17=[]
    for ds,folder in [('enzymes','hctf_safe_blend_all17_extrap6_prefix2_test'),('proteins','hctf_safe_blend_all17_proteins_portfolio4_extrap6_prefix2_test')]:
        root=ROOT/'results'/folder;cfg=json.loads(text(root/'run_config.json'));s=pd.read_csv(root/f'{ds}_selection.csv');all17.append(dict(dataset=ds,records=len(s),accepted=int(s.accepted.sum()),acceptance_rate=float(s.accepted.mean()),candidates=len(cfg['candidate_checkpoints'].split(';')),config=rel(root/'run_config.json'),config_sha256=sha(root/'run_config.json'),selection_sha256=sha(root/f'{ds}_selection.csv'),selection_metric_set=cfg['selection_metric_set'],alphas=cfg['alphas'],refinement=cfg['monotone_refinement_levels'],extrapolation_steps=cfg['monotone_extrapolation_steps'],prefix_alpha=cfg['monotone_extension_prefix_alpha'],device=cfg['device'],deterministic=cfg['deterministic_eval'],split=cfg['split'],same_budget_as_10plus7=False))
    csvout('evidence/all17_record_recount.csv',all17)
    # Execute actual alignment utility on synthetic CPU coordinates, not on any graph split.
    source=ROOT/'train_cser_spectral_mog_end2end.py';align=functions(source,['mtpf_center_layout','mtpf_procrustes_align'],[],{'torch':torch});x=torch.tensor([[0.,0.],[2.,0.],[0.,1.],[.4,.3]],dtype=torch.float64);rot=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64);p=3*x@rot+torch.tensor([4.,9.]);out=align['mtpf_procrustes_align'](p,x);reflection=x.clone();reflection[:,0]*=-1;reflectout=align['mtpf_procrustes_align'](reflection,x)
    save('evidence/alignment_fixtures.json',dict(proper_similarity_recovered=torch.allclose(out,x,atol=1e-12,rtol=0),reflection_not_fitted_exactly=not torch.allclose(reflectout,x,atol=1e-10),rotation_translation_scale=True,reflection_permitted=False,source=rel(source),source_sha256=sha(source)))
    save('evidence/safeblend_input_hashes.json',hashes);save('evidence/SAFEBLEND_COMPLETE.json',dict(records=sum(x['records'] for x in counts),Full_records=sum(x['records'] for x in counts if x['variant']=='full_nodrop'),Full_accepted=sum(x['accepted'] for x in counts if x['variant']=='full_nodrop'),Node_accepted=sum(x['accepted'] for x in counts if x['variant']=='node_only_nodrop'),fixture_pass=all(x['pass_check'] for x in fixture),new_predictions=False,historical_all_candidate_replay=False,all17=all17));event('safeblend','80archived selection files recounted; true source selection-loop AST executed on7controlled boundary fixtures; no model or Test data execution',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_safeblend_records.py');print('SAFE audited',sum(x['records'] for x in counts))

if __name__=='__main__':main()
