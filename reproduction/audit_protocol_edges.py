"""All-17 settings and runtime contract, read-only archived configuration audit."""
from audit_common import *

def main():
    p=ROOT/'scripts/evaluate_hctf_safe_blend.py';tree=ast.parse(text(p));ns={}
    names={'METRICS','HOLDOUT_METRICS','ALL_METRICS','LOWER_IS_BETTER','SCORE_FLOORS','ABS_TOLERANCE'}
    chosen=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(p),'exec'),ns)
    rows=[];hashes={rel(p):sha(p)}
    for ds,folder in [('enzymes','hctf_safe_blend_all17_extrap6_prefix2_test'),('proteins','hctf_safe_blend_all17_proteins_portfolio4_extrap6_prefix2_test')]:
        cfgpath=ROOT/'results'/folder/'run_config.json';cfg=json.loads(text(cfgpath));assert cfg['selection_metric_set']=='all'
        for m in ns['ALL_METRICS']:
            rows.append(dict(protocol=f'All-17 {ds}',metric=m,orientation='lower' if m in ns['LOWER_IS_BETTER'] else 'higher',absolute_tolerance=ns['ABS_TOLERANCE'][m],relative_tolerance=cfg['relative_tolerance'],score_floor=ns['SCORE_FLOORS'][m],weight=1.,online_audited=True,evaluation_only=False,config=rel(cfgpath),config_sha256=sha(cfgpath),source=rel(p),source_sha256=sha(p),historical_source_byte_binding='not included in run_config; current source/config agreement only'))
        for q in (ROOT/'results'/folder).glob('*.csv'):hashes[rel(q)]=sha(q)
        hashes[rel(cfgpath)]=sha(cfgpath)
    csvout('evidence/all17_metric_settings.csv',rows)
    runtimepath=ROOT/'results/hctf_end_to_end/COMPLETE.json';runtime=json.loads(text(runtimepath))
    hashes[rel(runtimepath)]=sha(runtimepath)
    for name in ['scripts/benchmark_hctf_end_to_end.py','results/hctf_end_to_end/end_to_end_observations.csv','scripts/analyze_hctf_submission_results.py','scripts/analyze_hctf_q1_controls.py']:
        hashes[name]=sha(ROOT/name)
    save('evidence/runtime_contract.json',dict(**runtime,source=rel(ROOT/'scripts/benchmark_hctf_end_to_end.py'),source_sha256=sha(ROOT/'scripts/benchmark_hctf_end_to_end.py'),actual_rebenchmark=False,batch_size_inferred_from_code=1,synchronization_source_lines='27-29,85,93,95,101,120',baseline_audit='Includes record construction/community detection and baseline measure_layout; no nonzero trials for baseline',tie_rule_scope='timing harness only strict score > best+1e-12; not the full paired-selector near-tie loop'))
    save('evidence/protocol_edge_input_hashes.json',hashes)
    event('protocol_edges','Read both All17 run_config metric budgets and17metric settings; bound archived runtime contract; no timing or model inference performed',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_protocol_edges.py')
    print('ALL17 settings',len(rows),'runtime',runtime['configuration'],runtime['device'])

if __name__=='__main__':main()
