"""Recheck all ten mechanism endpoints using saved committed rows only."""
from audit_common import *
import importlib.util

def main():
    spec=importlib.util.spec_from_file_location('audit_stat',HERE/'02_BOOTSTRAP_AUDIT.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);ns=a.original_analysis();archive=pd.read_csv(ROOT/'results/hctf_mechanistic_counterfactuals_analysis/crossed_bootstrap_ci.csv');frames={};tables=[]
    for ds in ['enzymes','fb15k237','sbm']:
        for seed in [17,42,73]:frames[ds,seed]=pd.read_csv(ROOT/f'results/hctf_mechanistic_counterfactuals/strict_{ds}_s{seed}/observations.csv')
    for ref in ['same_weight_gate_zero','same_weight_gate_one','top_down_only','bottom_up_only']:
        pairs={k:a.join(f[(f.source==ref)&(f.stage=='raw')],f[(f.source=='full')&(f.stage=='raw')],['graph_name','batch_id','local_graph_id'],'graph_name') for k,f in frames.items()};calc=ns['hierarchical_bootstrap'](pairs,4000,20261009,metrics=ns['CORE_METRICS']);label='full_raw_vs_'+ref+'_raw';calc['comparison']=label;saved=archive[archive.comparison==label];j=calc.merge(saved,on=['dataset','metric','comparison'],suffixes=('_recomputed','_archived'),validate='one_to_one')
        for field in ['mean_gain_pct','ci_low_pct','ci_high_pct']:assert (j[field+'_recomputed']-j[field+'_archived']).abs().max()<1e-9
        tables.append(j);csvout('evidence/mechanism_all_metric_reproduction.csv',pd.concat(tables));print('MECHANISM all10',ref,flush=True)
    save('evidence/MECHANISM_METRICS_COMPLETE.json',dict(rows=sum(len(t) for t in tables),all_original_values_match=True,models_executed=False));event('statistics','160rawmechanism/domain/endpoint rows independently reproduced, including community ratio intervals',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_mechanism_all_metrics.py')

if __name__=='__main__':main()
