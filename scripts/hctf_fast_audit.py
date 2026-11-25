"""Chunked NumPy audit kernels; semantics match the recorded Python evaluator.

This reduces interpreter overhead, not the quadratic edge-pair complexity.
Training snapshots are deliberately unaffected; only new paired/isolated
evaluation tools install these tested kernels.
"""
import numpy as np
import evaluate_extra_metrics as original


def _ccw(a,b,c):
    return (c[...,1]-a[...,1])*(b[...,0]-a[...,0]) > (b[...,1]-a[...,1])*(c[...,0]-a[...,0])


def _crossing_chunks(pos, edge_index, block=128):
    edges=np.asarray(original.unique_undirected_edges(edge_index),dtype=np.int64).reshape(-1,2)
    for start in range(0,len(edges),block):
        first=edges[start:start+block]
        u,v=first[:,0,None],first[:,1,None]
        x,y=edges[None,:,0],edges[None,:,1]
        valid=(np.arange(start,start+len(first))[:,None] < np.arange(len(edges))[None,:])
        valid &= (u!=x)&(u!=y)&(v!=x)&(v!=y)
        a,b,c,d=pos[u],pos[v],pos[x],pos[y]
        crossing=(_ccw(a,c,d)!=_ccw(b,c,d))&(_ccw(a,b,c)!=_ccw(a,b,d))&valid
        yield valid,crossing,b-a,d-c


def edge_crossing_count(pos,edge_index):
    # Preserve input float32/float64 arithmetic exactly as the old ccw kernel.
    positions=pos.detach().cpu().numpy()
    checked=crossings=0
    for valid,crossing,_,_ in _crossing_chunks(positions,edge_index):
        checked+=int(valid.sum()); crossings+=int(crossing.sum())
    return crossings,crossings/checked if checked else 0.


def crossing_angle_resolution(pos,edge_index):
    positions=pos.detach().cpu().numpy().astype(np.float64,copy=False)
    angles=[]
    for _,crossing,first,second in _crossing_chunks(positions,edge_index):
        denominator=np.linalg.norm(first,axis=-1)*np.linalg.norm(second,axis=-1)
        valid=crossing&(denominator>1e-12)
        dot=np.sum(first*second,axis=-1)
        cosine=np.clip(np.abs(dot[valid])/denominator[valid],0.,1.)
        angles.extend(np.degrees(np.arccos(cosine)).tolist())
    return (float(np.mean(angles)),float(np.percentile(angles,5))) if angles else (90.,90.)


def node_edge_clearance(pos,edge_index):
    positions=pos.detach().cpu().numpy().astype(np.float64,copy=False)
    edges=np.asarray(original.unique_undirected_edges(edge_index),dtype=np.int64).reshape(-1,2)
    if not len(edges): return 1.,1.
    start,end=positions[edges[:,0]],positions[edges[:,1]]
    segment=end-start
    denominator=np.sum(segment*segment,axis=1)
    scale=max(float(np.median(np.linalg.norm(segment,axis=1))),1e-12)
    values=[]
    for offset in range(0,len(positions),128):
        points=positions[offset:offset+128]
        displacement=points[:,None,:]-start[None,:,:]
        fraction=np.sum(displacement*segment[None,:,:],axis=-1)/np.maximum(denominator,1e-18)
        fraction=np.clip(fraction,0.,1.)
        fraction[:,denominator<=1e-18]=0.
        projection=start[None,:,:]+fraction[:,:,None]*segment[None,:,:]
        distance=np.linalg.norm(points[:,None,:]-projection,axis=-1)
        ids=np.arange(offset,offset+len(points))[:,None]
        incident=(ids==edges[None,:,0])|(ids==edges[None,:,1])
        distance[incident]=np.inf
        minimum=distance.min(axis=1)
        values.extend((minimum[np.isfinite(minimum)]/scale).tolist())
    return (float(np.mean(values)),float(np.percentile(values,5))) if values else (1.,1.)


def install_fast_audit(evaluator):
    evaluator.edge_crossing_count=edge_crossing_count
    original.crossing_angle_resolution=crossing_angle_resolution
    original.node_edge_clearance=node_edge_clearance
