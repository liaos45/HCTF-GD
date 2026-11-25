"""Archived identities/manifests and local source provenance, without dataset loaders."""
from audit_common import *
import tarfile
from types import SimpleNamespace
import networkx as nx


def main():
    meta=pd.read_csv(ROOT/'results/hctf_graph_metadata.csv');ex=pd.read_csv(ROOT/'results/hctf_warm_start_exposure/per_graph_exposure.csv');spl=pd.read_csv(ROOT/'results/hctf_warm_start_exposure/split_exposure.csv')
    merge=meta.merge(ex,on=['dataset','seed','test_graph_id','original_graph_index','graph_name'],validate='one_to_one');assert len(merge)==len(meta)==9725
    counts=ex.groupby(['dataset','seed','recorded_warm_split']).size().unstack(fill_value=0);reconstructed=[]
    for x in spl.itertuples():
        r=counts.loc[(x.dataset,x.seed)];available=bool(x.warm_split_reconstruction_available)
        if available:
            assert int(r.get('train',0))==x.warm_train_overlap and int(r.get('validation',0))==x.warm_validation_overlap and int(r.get('test',0))==x.warm_test_overlap
        reconstructed.append(dict(dataset=x.dataset,seed=x.seed,records=x.test_graphs,reconstructable=available,warm_train_overlap=int(r.get('train',0)),warm_validation_overlap=int(r.get('validation',0)),warm_test_overlap=int(r.get('test',0)),unknown=int(r.get('unknown',0)),warm_checkpoint=x.warm_checkpoint,identity_basis='recorded original index + graph_name + order/num_nodes/num_edges; not historical cryptographic edge fingerprints'))
    csvout('evidence/warm_overlap_recount.csv',reconstructed);c=pd.DataFrame(reconstructed);summary=dict(total_downstream_splits=len(c),reconstructed=int(c.reconstructable.sum()),train_overlap=int((c.warm_train_overlap>0).sum()),reconstructed_without_train_overlap=int((c.reconstructable&(c.warm_train_overlap==0)).sum()),unknown=int((~c.reconstructable).sum()),zero_train_overlap_all_warm_test=bool((c[c.reconstructable&(c.warm_train_overlap==0)].warm_test_overlap==c[c.reconstructable&(c.warm_train_overlap==0)].records).all()),test_records=len(ex),distinct_domain_graph_ids=ex[['dataset','original_graph_index']].drop_duplicates().shape[0]);save('evidence/WARM_OVERLAP_SUMMARY.json',summary)
    # Reconstruct recorded random warm memberships using ONLY saved args and index
    # strings. This executes the real split arithmetic without a dataset/Test loader.
    source=ROOT/'train_cser_spectral_mog_end2end.py';tree=ast.parse(text(source));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_graph_dataset');env={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),env)
    arguments=json.loads(text(HERE/'evidence/actual_checkpoint_args.json'));independent=[]
    for ds in sorted(ex.dataset.unique()):
        warm=next(q for q in arguments if q['cohort']=='legacy' and q['dataset']==ds and q['variant']=='warm');a=warm['args']
        if not all(k in a for k in ['dataset','seed','index_file','layouts_file','batch_size']):
            independent.append(dict(dataset=ds,status='UNVERIFIED',reason='saved warm checkpoint lacks required split arguments',graphs=0));continue
        p=Path(a['index_file']);p=p if p.is_absolute() else ROOT/p;names=[v.strip() for v in text(p).splitlines() if v.strip()];train,val,held=env['split_graph_dataset'](names,SimpleNamespace(**a));membership={v:label for label,items in [('train',train),('validation',val),('test',held)] for v in items};subset=ex[ex.dataset==ds];actual=[membership.get(name,'missing') for name in subset.graph_name];assert np.array_equal(actual,subset.recorded_warm_split.to_numpy())
        independent.append(dict(dataset=ds,status='VERIFIED recorded-index membership',reason='saved checkpoint args + actual split function on hashed index strings; no graph loader; historical topology byte identity remains separate',graphs=len(subset),train=len(train),validation=len(val),test=len(held),index=rel(p),index_sha256=sha(p),warm_checkpoint=warm['checkpoint'],warm_sha256=warm['sha256']))
    csvout('evidence/warm_index_membership_reconstruction.csv',independent)
    sources={
      'enzymes':('deepgd/datasets/tu_dataset.py','TUDataset ENZYMES; preprocessing/subset see filtered index'),
      'proteins':('deepgd/datasets/tu_dataset.py','TUDataset PROTEINS; filtered processed subset'),
      'imdb_binary':('deepgd/datasets/tu_dataset.py','TUDataset IMDB-BINARY; collaboration/movie graph corpus'),
      'sbm':('deepgd/datasets/sbm_dataset.py','networkx stochastic block generator, seed42 by saved construction defaults; filtered index controls actual population'),
      'wn18rr':('deepgd/datasets/kg_sampled_dataset.py','Sampled induced/BFS subgraphs of WN18RR training triples; not full benchmark transductive evaluation'),
      'fb15k237':('deepgd/datasets/kg_sampled_dataset.py','Sampled induced/BFS subgraphs of FB15K237 training triples; not full benchmark entity prediction'),
      'north':('deepgd/datasets/north_dataset.py','AT&T/Graph Catalog directed GraphML archive north-graphml.tgz; distinct from ROME; converted to undirected graph drawing inputs'),
      'rome':('deepgd/datasets/rome_dataset.py','GDToolkit/Rome undirected GraphML corpus; canonical subset split if >=11000graphs')}
    rows=[];indexhash={}
    for ds,(src,description) in sources.items():
        p=ROOT/src;index=ROOT/f'assets/{ds}_index.txt';names=[x.strip() for x in text(index).splitlines() if x.strip()];m=meta[meta.dataset==ds];src_text=text(p);urls=re.findall(r'https?://[^\s\"\'<>]+',src_text);indexhash[rel(index)]=sha(index)
        rows.append(dict(dataset=ds,source_file=src,source_sha256=sha(p),recorded_origin=description,URLs=';'.join(dict.fromkeys(urls)),filtered_pool_index=rel(index),filtered_pool_graphs=len(names),index_sha256=sha(index),archived_evaluation_node_min=int(m.num_nodes.min()),archived_evaluation_node_max=int(m.num_nodes.max()),archived_evaluation_edge_min=int(m.num_edges.min()),archived_evaluation_edge_max=int(m.num_edges.max()),range_scope='archived downstream evaluation rows, NOT entire original collection',community='training greedy with saved community_seed42; SBM planted reference used in some strict evaluation, not universally',splits='legacy seeded random80/10/10 except Rome canonical; strict size mode separate,see actual scripts/markers',historical_download_version='exact original download date/version not saved',graph_identity_hash_status='original-index/name/order validation; no all-row historical edge+partition hashes',source_status='CODE-VERIFIED ONLY; archive index read and rehashed'))
    csvout('05_DATASET_PROVENANCE.csv',rows)
    archive=ROOT/'north-graphml.tgz'
    if archive.exists():
        with tarfile.open(archive,'r:gz') as t:
            members=t.getnames();logs=[n for n in members if n.endswith('Graph.log')];log=t.extractfile(logs[0]).read().decode('utf-8',errors='replace') if logs else '';write('evidence/NORTH_Archive_Graph_log.txt',log)
            files=[n for n in members if n.endswith('.graphml')];names={Path(n).stem for n in files};sample=t.extractfile(files[0]).read().decode();parsed=nx.parse_graphml(sample)
            indexnames=set(text(ROOT/'assets/north_index.txt').splitlines());assert names==indexnames
            save('evidence/NORTH_ARCHIVE_METADATA.json',dict(archive=rel(archive),sha256=sha(archive),members=len(members),GraphML_members=len(files),all_index_names_match_archive=True,Graph_log=logs,log_origin_directed=True,sample_graph=files[0],sample_has_edgedefault='edgedefault=' in sample,NetworkX_sample_read_as_directed=parsed.is_directed(),direction_note='Original collection log says directed; legacy GraphML lacks edgedefault and is parsed as undirected, then NormalizeGraph represents it with reciprocal directed edges. No directed-layout guarantee is made.',source_page='https://graphdrawing.unipg.it/data.html',source_archive_url='https://graphdrawing.unipg.it/data/north-graphml.tgz',historical_download_byte_binding=False))
    # Safe record identities can be reconstructed by matching archived order/size,not by filenames alone.
    checks=[]
    for (ds,seed),m in meta.groupby(['dataset','seed']):
        m=m.sort_values('test_graph_id')
        for v in ['baseline','full_nodrop','node_only_nodrop']:
            f=pd.read_csv(ROOT/f'results/hctf_submission_main/submission_{ds}_{v}_s{seed}_test.csv').sort_values(['batch_id','local_graph_id']);same=np.array_equal(f[['num_nodes','num_edges']].to_numpy(),m[['num_nodes','num_edges']].to_numpy());assert same;checks.append(dict(dataset=ds,seed=seed,variant=v,rows=len(f),metadata_order_n_m_match=same,original_name_binding='archived metadata, full historical topology hashes missing'))
    csvout('evidence/main_metadata_pairing_checks.csv',checks);save('evidence/dataset_input_hashes.json',{rel(p):sha(p) for p in [ROOT/'results/hctf_graph_metadata.csv',ROOT/'results/hctf_warm_start_exposure/per_graph_exposure.csv',ROOT/'results/hctf_warm_start_exposure/split_exposure.csv',*[(ROOT/k) for k in indexhash]]});event('datasets',f'Recounted9725archived graph/seed memberships; {summary}; no loader or graph prediction executed',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_dataset_records.py');print(summary)

if __name__=='__main__':main()
