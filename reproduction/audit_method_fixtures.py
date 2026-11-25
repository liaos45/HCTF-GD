"""CPU, synthetic mechanism fixtures from sealed historical source. No graph loaders."""
from audit_common import *
import torch
from collections import defaultdict,deque,OrderedDict
from types import SimpleNamespace
import typing
import networkx as nx


def main():
    src=ROOT/'results/hctf_strict_size_ood/source_snapshot_strict/train_cser_spectral_mog_end2end.py';tree=ast.parse(text(src));selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['build_caid_context','cpsg_loss','hctf_exact_risk_targets']]
    names={'HCTF_EXACT_METRICS','HCTF_EXACT_FLOORS','HCTF_EXACT_HIGHER_IS_BETTER'}
    selected=[n for n in tree.body if n in selected or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    ns={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),str(src),'exec'),ns)
    aedges=torch.tensor([[0,1,2,3],[1,0,3,2]],dtype=torch.long);bedges=torch.tensor([[4,5],[5,4]],dtype=torch.long);edges=torch.cat([aedges,bedges],1);labels=torch.tensor([0,0,1,1,0,1]);batch=torch.tensor([0,0,0,0,1,1]);pi=torch.tensor([[0],[2]],dtype=torch.long);ap=torch.tensor([2.])
    saved=next(q for q in json.loads(text(HERE/'evidence/actual_checkpoint_args.json')) if q['cohort']=='strict' and q['dataset']=='enzymes' and q['seed']==17 and q['variant']=='Full')
    kwargs={k:saved['args']['caid_'+k] for k in ['intra_shrink','adj_expand','far_expand','min_ratio','max_ratio']}
    one=ns['build_caid_context'](ap,pi,aedges,labels[:4],batch[:4],**kwargs);combined=ns['build_caid_context'](ap,pi,edges,labels,batch,**kwargs);unique=ns['build_caid_context'](ap,pi,edges,torch.tensor([0,0,1,1,2,3]),batch,**kwargs)
    caid=dict(snapshot=rel(src),snapshot_sha256=sha(src),same_A_node_pair=[0,2],A_alone_far=float(one['far_pair'][0]),A_alone_ratio=float(one['ratio'][0]),A_with_other_graph_local_labels_reused_adjacent=float(combined['adjacent_pair'][0]),A_with_other_graph_ratio=float(combined['ratio'][0]),A_with_graph_qualified_labels_ratio=float(unique['ratio'][0]),expected_per_graph_invariance_broken=not torch.equal(one['ratio'],combined['ratio']))
    args=SimpleNamespace(**saved['args'])
    pos=torch.tensor([[0.,0.],[.2,0.],[2.,0.],[2.2,0.],[0.,1.],[1.,1.]])
    baseline=ns['cpsg_loss'](pos,edges,labels,batch,args);shifted=pos.clone();shifted[4:]+=100;after=ns['cpsg_loss'](shifted,edges,labels,batch,args);qlabel=torch.tensor([0,0,1,1,2,3]);qbefore=ns['cpsg_loss'](pos,edges,qlabel,batch,args);qafter=ns['cpsg_loss'](shifted,edges,qlabel,batch,args)
    ma=dict(legacy_before=[float(v) for v in baseline],legacy_after_other_graph_translation=[float(v) for v in after],qualified_before=[float(v) for v in qbefore],qualified_after=[float(v) for v in qafter],legacy_translation_invariance_broken=not torch.allclose(baseline[0],after[0]),qualified_proto_invariant=torch.allclose(qbefore[0],qafter[0],atol=1e-6))
    assert caid['expected_per_graph_invariance_broken'] and ma['legacy_translation_invariance_broken']
    save('evidence/graph_local_community_counterexamples.json',dict(CAID=caid,MA_CPSG=ma,fixture_scope='synthetic tensors only, no trained model/data split; no change to historical algorithm'))
    # Prove balanced BCE differs from pooled BCE when class sizes differ.
    risk=torch.tensor([.5,.5,.5,.5]);target=torch.tensor([1.,1.,1.,.1]);safe=torch.tensor([True,True,True,False]);bce=torch.nn.functional.binary_cross_entropy(risk,target,reduction='none');balanced=.5*bce[safe].mean()+.5*bce[~safe].mean()
    risk2=torch.tensor([.9,.8,.7,.8]);bce2=torch.nn.functional.binary_cross_entropy(risk2,target,reduction='none');save('evidence/gate_reduction_fixture.json',dict(three_safe_one_unsafe=True,balanced_group_BCE=float(.5*bce2[safe].mean()+.5*bce2[~safe].mean()),pooled_BCE=float(bce2.mean()),source=rel(src),line_start=1856,line_end=1911))
    learned=torch.tensor([.8,.8],requires_grad=True);teacher=torch.tensor([[1.,.5,.4],[1.,.5,.4]]);candidate=torch.tensor([[.9,.49,.41],[1.1,.51,.39]])
    target2,regret,safe2=ns['hctf_exact_risk_targets'](learned,candidate,teacher,[0,4,9],0.,.1);loss=torch.nn.functional.binary_cross_entropy(learned.clamp(1e-6,1-1e-6),target2.detach(),reduction='none');balanced2=.5*loss[safe2].mean()+.5*loss[~safe2].mean();balanced2.backward();save('evidence/gate_target_direction_gradient_fixture.json',dict(target=target2.detach().tolist(),max_regret=regret.tolist(),safe=safe2.tolist(),target_requires_grad=target2.requires_grad,gate_gradients=learned.grad.tolist(),finite_gradients=bool(torch.isfinite(learned.grad).all()),teacher_metrics_receive_no_grad=not teacher.requires_grad,Stress_definition='HCTF_EXACT_METRICS[0]=batch_stress pair-sum, floor .05',edge_CV_floor=.05,silhouette_floor=.05,synthetic_only=True,optimizer_steps=0))
    # Execute the actual forest constructor (not a beautified hand-picked forest).
    fs=ROOT/'results/hctf_strict_size_ood/source_snapshot_strict/deepgd/model/generator/hctf.py';module=ast.parse(text(fs));clazz=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='CoverageAwareTreeCover');classns={'OrderedDict':OrderedDict,'defaultdict':defaultdict,'deque':deque,'torch':torch,'LongTensor':torch.LongTensor,'FloatTensor':torch.FloatTensor,'TreeCoverBatch':None,**vars(typing)};exec(compile(ast.Module(body=[clazz],type_ignores=[]),str(fs),'exec',flags=__future_annotations()),classns)
    # Two components plus an isolated vertex ensure forest, not single-tree assumptions.
    edgeset=((0,1),(0,2),(1,2),(1,3),(2,3),(4,5),(4,6),(5,6));cover=classns['CoverageAwareTreeCover'](num_trees=3,max_cache_size=0);out=cover._build_local_cover(8,edgeset);trees=[];G=nx.Graph();G.add_nodes_from(range(8));G.add_edges_from(edgeset)
    for k in range(3):
        selected=set(tuple(sorted((u,v))) for u,v,t in zip(out[0],out[1],out[3]) if t==k);F=nx.Graph();F.add_nodes_from(range(8));F.add_edges_from(selected);assert selected.issubset(edgeset) and nx.is_forest(F) and {frozenset(c) for c in nx.connected_components(F)}=={frozenset(c) for c in nx.connected_components(G)}
        trees.append(dict(tree_id=k,edges=sorted(selected),edge_count=len(selected),acyclic=True,components_preserved=True,new_vs_previous=len(selected-set(e for t in trees for e in map(tuple,t['edges'])))))
    save('evidence/forest_algorithm_fixture.json',dict(source=rel(fs),sha256=sha(fs),synthetic_fixture=True,nodes=8,original_edges=edgeset,trees=trees,deterministic=cover._build_local_cover(8,edgeset)==out,not_manuscript_figure_example=True))
    event('method','Executed sealed-source CAID/MA graph-local counterexamples, gate-reduction fixture and actual union-find forest constructor on synthetic tensors/graph only',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_method_fixtures.py');print('METHOD source fixtures complete')

def __future_annotations():
    import __future__
    return __future__.annotations.compiler_flag

if __name__=='__main__':main()
