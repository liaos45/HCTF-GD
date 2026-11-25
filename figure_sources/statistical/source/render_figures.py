"""Reproducible real-result figures at 180 mm; no model calls or fitting.

Inputs: prepared coordinate/edge/metric tables and verified statistical
archives. Display transforms are proper rigid rotation + translation and one
common isotropic scale per graph. Metrics are never computed on display data.
All points and negative/uncertain effects are retained.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "source_data"
WIDTH_MM = 180.0
INK = "#263343"; GRAY = "#75818F"; EDGE = "#AEB8C2"
BLUE = "#638FB5"; GREEN = "#689C83"; PURPLE = "#8977B2"; ORANGE = "#C89561"
PALETTE = [BLUE, GREEN, PURPLE, ORANGE, "#8399A2", "#B18D9C", "#9AA570", "#7D92AD"]
DS = ["enzymes","fb15k237","imdb_binary","north","proteins","rome","sbm","wn18rr"]
NAMES = {"enzymes":"ENZYMES","fb15k237":"FB15K237","imdb_binary":"IMDB-BINARY","north":"NORTH",
         "proteins":"PROTEINS","rome":"ROME","sbm":"SBM","wn18rr":"WN18RR","ALL":"Equal-domain mean"}
METHODS = ["Matching Backbone","Node-only HCTF","Raw Full HCTF","SafeBlend Final"]
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
                    "font.size":8.2,"axes.labelsize":8.2,"axes.titlesize":9.2,"xtick.labelsize":8.0,
                    "ytick.labelsize":8.0,"legend.fontsize":8.0,"axes.linewidth":.65,
                    "pdf.fonttype":42,"svg.fonttype":"none","figure.facecolor":"white",
                    "savefig.facecolor":"white","axes.facecolor":"white"})


def read(name): return pd.read_csv(DATA / name)


def figure(height):
    return plt.figure(figsize=(WIDTH_MM/25.4,height/25.4),dpi=600)


def text_mm(fig,x,y,text,size=8.2,**kwargs):
    h=fig.get_figheight()*25.4
    return fig.text(x/WIDTH_MM,y/h,text,fontsize=size,color=INK,ha=kwargs.pop("ha","left"),
                    va=kwargs.pop("va","center"),**kwargs)


def axis_mm(fig,x,y,w,h):
    height=fig.get_figheight()*25.4
    return fig.add_axes([x/WIDTH_MM,y/height,w/WIDTH_MM,h/height])


def clean(ax,grid="x"):
    for side in ("top","right"): ax.spines[side].set_visible(False)
    if grid: ax.grid(axis=grid,color="#E5E9ED",lw=.5,zorder=0)
    ax.tick_params(length=2.5,width=.6)


def save(fig,stem):
    for folder in ("pdf","svg","png","preview"): (OUT/folder).mkdir(exist_ok=True)
    # Report actual figure-bound violations; no bbox='tight' size changes.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); bbox=fig.bbox
    inactive_ticks=set()
    # Matplotlib creates tick objects just outside the view interval; their
    # labels are not painted and must not be mistaken for clipped page text.
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            low,high=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-12 <= tick.get_loc() <= high+1e-12:
                    inactive_ticks.update([tick.label1,tick.label2])
    outside=[]
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text() or artist in inactive_ticks: continue
        extent=artist.get_window_extent(renderer)
        if extent.x0 < -1 or extent.y0 < -1 or extent.x1 > bbox.x1+1 or extent.y1 > bbox.y1+1:
            outside.append(artist.get_text())
    if outside: raise ValueError(f"Out-of-page text in {stem}: {outside}")
    fig.savefig(OUT/"pdf"/f"{stem}.pdf",facecolor="white")
    fig.savefig(OUT/"svg"/f"{stem}.svg",facecolor="white")
    png=OUT/"png"/f"{stem}.png"
    fig.savefig(png,dpi=600,facecolor="white")
    with Image.open(png) as im:
        rgb=im.convert("RGB"); rgb.save(png,dpi=(600,600))
        w=round(180/25.4*300)
        rgb.resize((w,round(rgb.height*w/rgb.width)),Image.Resampling.LANCZOS).save(
            OUT/"preview"/f"{stem}_180mm.png",dpi=(300,300))
    plt.close(fig)
    print("[RENDERED]",stem,flush=True)


def ci(ax,mean,low,high,y,color,marker="o",label=None):
    # Percentile intervals are drawn independently of the point estimate.
    ax.hlines(y,low,high,color=color,lw=1.0,zorder=2)
    ax.plot([low,low],[y-.055,y+.055],color=color,lw=.8)
    ax.plot([high,high],[y-.055,y+.055],color=color,lw=.8)
    ax.plot(mean,y,marker,color=color,ms=3.8,label=label,zorder=3)


def rigid_align(source,reference):
    s=source-source.mean(0); r=reference-reference.mean(0)
    u,_,vt=np.linalg.svd(s.T@r); rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1; rotation=u@vt
    return s@rotation,rotation


def case_data(seed,gid):
    nodes=read("layout_nodes.csv"); nodes=nodes[(nodes.seed==seed)&(nodes.graph_id==gid)]
    edges=read("layout_edges.csv"); edges=edges[(edges.seed==seed)&(edges.graph_id==gid)][["source","target"]].to_numpy(int)
    positions={m:nodes[nodes.method==m].sort_values("node_id")[["x","y"]].to_numpy() for m in METHODS}
    labels=nodes[nodes.method==METHODS[0]].sort_values("node_id").community.to_numpy(int)
    for method in METHODS:
        group=nodes[nodes.method==method].sort_values("node_id")
        assert group.node_id.tolist()==list(range(len(labels)))
        assert np.array_equal(group.community.to_numpy(int),labels)
    aligned={}; transforms=[]
    reference=positions[METHODS[0]]
    for method,xy in positions.items():
        rotated,rotation=rigid_align(xy,reference)
        aligned[method]=rotated
        transforms.append(dict(seed=seed,graph_id=gid,method=method,center_x=xy.mean(0)[0],center_y=xy.mean(0)[1],
                               r00=rotation[0,0],r01=rotation[0,1],r10=rotation[1,0],r11=rotation[1,1]))
    scale=max(float(np.max(np.abs(xy))) for xy in aligned.values())*1.08
    for method in aligned: aligned[method]=aligned[method]/scale
    for row in transforms: row["shared_display_scale"]=scale
    metrics=read("layout_metrics.csv")
    metrics=metrics[(metrics.seed==seed)&(metrics.graph_id==gid)].set_index("method")
    return aligned,edges,labels,metrics,transforms


def draw_graph(ax,positions,edges,labels,boundary=False,subset=None):
    visible=np.arange(len(labels)) if subset is None else np.asarray(subset,int)
    ids=set(visible.tolist())
    kept=np.array([e for e in edges if int(e[0]) in ids and int(e[1]) in ids],int).reshape(-1,2)
    if len(kept): ax.add_collection(LineCollection(positions[kept],colors="#A6AFB9",linewidths=.44,alpha=.67,zorder=1))
    colors=[PALETTE[int(labels[i])%len(PALETTE)] for i in visible]
    ax.scatter(positions[visible,0],positions[visible,1],s=13,c=colors,edgecolors="white",linewidths=.28,zorder=3)
    if boundary:
        b=set(int(i) for a,c in edges if labels[a]!=labels[c] for i in (a,c))
        selected=[i for i in visible if i in b]
        if selected: ax.scatter(positions[selected,0],positions[selected,1],s=23,facecolors="none",edgecolors=ORANGE,linewidths=.55,zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    for spine in ax.spines.values(): spine.set_visible(False)


def qualitative():
    fig=figure(173); transformations=[]
    column=[24.5,68,111.5,155]
    for x,title in zip(column,["Matching\nBackbone","Node-only HCTF","Raw Full HCTF","SafeBlend Final"]):
        text_mm(fig,x,168,title,8.8,ha="center",weight="bold")
    cases=[(17,95,"Main-protocol fallback; legacy partition sensitivity"),(17,13,"Raw failure; exact fallback"),
           (73,9,"Accepted; archived Stress-gain rank 1")]
    for row,(seed,gid,role) in enumerate(cases):
        y=156-row*51
        text_mm(fig,4,y,f"SBM | seed {seed} | graph {gid} | {role}",8.4,weight="bold")
        positions,edges,labels,metrics,trans=case_data(seed,gid); transformations+=trans
        for x,method in zip(column,METHODS):
            ax=axis_mm(fig,x-20,y-38,40,34)
            draw_graph(ax,positions[method],edges,labels)
            m=metrics.loc[method]
            text_mm(fig,x,y-43.5,f"S {m.stress_normalized:.4f} | X {m.edge_crossing:.0f}\nSep/Comp {m.community_sep_comp_ratio:.3f}",
                    8.0,ha="center")
    text_mm(fig,90,2.6,"S: pair-normalized Stress; X: crossings; higher Sep/Comp is better",8.0,ha="center")
    pd.DataFrame(transformations).to_csv(DATA/"display_transforms.csv",index=False)
    save(fig,"qualitative_layout_comparison")


def community_readability():
    fig=figure(151); centers=[31,90,149]; methods=[METHODS[0],METHODS[2],METHODS[3]]; ego_rows=[]
    for x,method in zip(centers,methods): text_mm(fig,x,146,method,9.0,ha="center",weight="bold")
    for row,(seed,gid,description) in enumerate([(73,103,"Accepted: community geometry improves"),(17,13,"Raw community degradation; exact fallback")]):
        y=133-row*65
        positions,edges,labels,metrics,_=case_data(seed,gid)
        graph=nx.Graph(); graph.add_nodes_from(range(len(labels))); graph.add_edges_from(edges)
        candidates=[i for i in graph if any(labels[i]!=labels[j] for j in graph[i])]
        root=sorted(candidates,key=lambda i:(-sum(labels[i]!=labels[j] for j in graph[i]),-graph.degree(i),i))[0]
        subset=sorted({root}|set(graph[root]))
        zoom={m:positions[m][subset]-positions[m][subset].mean(0) for m in methods}
        zoom_scale=max(float(np.max(np.abs(z))) for z in zoom.values())*1.12
        text_mm(fig,4,y,f"SBM | seed {seed} | graph {gid} | {description}",8.4,weight="bold")
        for x,method in zip(centers,methods):
            ax=axis_mm(fig,x-25,y-36,50,32); draw_graph(ax,positions[method],edges,labels,boundary=True)
            m=metrics.loc[method]
            text_mm(fig,x,y-43,f"C {m.community_compactness:.3f} | Sep/C {m.community_sep_comp_ratio:.3f}\nClearance {m.node_edge_clearance:.3f}",8.0,ha="center")
            zpos=positions[method]-positions[method][subset].mean(0)
            zpos=zpos/max(zoom_scale,1e-12)
            za=axis_mm(fig,x-12,y-61,24,12)
            draw_graph(za,zpos,edges,labels,boundary=True,subset=subset)
            for side in za.spines.values(): side.set_visible(True); side.set_color(EDGE); side.set_linewidth(.4)
            ego_rows.extend(dict(seed=seed,graph_id=gid,ego_root=root,node_id=int(i),method=method) for i in subset)
    pd.DataFrame(ego_rows).to_csv(DATA/"boundary_ego_nodes.csv",index=False)
    text_mm(fig,90,2.3,"C: compactness (lower better); orange rings: boundary nodes; insets: same ego IDs",8.0,ha="center")
    save(fig,"community_readability_cases")


def paired_structure():
    fig=figure(106)
    text_mm(fig,4,100,"(a) Shared warm-start: eight domains",9.1,weight="bold")
    text_mm(fig,107,100,"(b) Strict size-OOD",9.1,weight="bold")
    ax=axis_mm(fig,28,24,66,67)
    data=read("main_confidence_archive.csv"); data=data[data.metric=="graph_stress"]
    order=DS+["ALL"]
    for i,ds in enumerate(order):
        yy=len(order)-1-i
        for comp,color,marker,offset in [("full_vs_node",BLUE,"o",.14),("full_vs_backbone",GREEN,"s",-.14)]:
            row=data[(data.dataset==ds)&(data.comparison==comp)].iloc[0]
            ci(ax,row.mean_gain_pct,row.ci_low_pct,row.ci_high_pct,yy+offset,ORANGE if ds=="ALL" else color,marker)
    ax.axvline(0,color=GRAY,lw=.7); ax.set_yticks(np.arange(9),[NAMES[d] for d in order][::-1])
    ax.set_xlim(-15,25); ax.set_ylim(-.6,8.6); ax.set_xlabel("Stress improvement (%)"); clean(ax)
    ax=axis_mm(fig,123,24,52,67)
    data=read("strict_confidence_archive.csv")
    data=data[(data.metric=="graph_stress")&(data.comparison=="full_raw_vs_node_raw")]
    order=["enzymes","fb15k237","sbm","ALL"]
    for i,ds in enumerate(order):
        row=data[data.dataset==ds].iloc[0]
        ci(ax,row.mean_gain_pct,row.ci_low_pct,row.ci_high_pct,3-i,ORANGE if ds=="ALL" else BLUE,"D" if ds=="ALL" else "o")
    ax.axvline(0,color=GRAY,lw=.7); ax.set_yticks(np.arange(4),[NAMES[d] for d in order][::-1]); ax.set_ylim(-.5,3.5)
    ax.set_xlim(-.16,1.23); ax.set_xlabel("Stress improvement (%)"); clean(ax)
    text_mm(fig,108,11,"FB15K237 point: +0.002446%",8.0)
    handles=[Line2D([],[],color=BLUE,marker="o",lw=1,label="Full vs Node-only"),Line2D([],[],color=GREEN,marker="s",lw=1,label="Full vs Matching Backbone")]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.05,.005),ncol=2,frameon=False)
    save(fig,"paired_structure_and_task")


def safe_holdout():
    fig=figure(111)
    text_mm(fig,4,105,"(a) 10+7 acceptance",9.2,weight="bold")
    text_mm(fig,94,105,"(b) Evaluation-only endpoints",9.2,weight="bold")
    ax=axis_mm(fig,22,31,47,55); coverage=read("safe_acceptance.csv").set_index("variant")
    for y,v,color in [(1,"full_nodrop",BLUE),(0,"node_only_nodrop",GREEN)]:
        r=coverage.loc[v]; ax.barh(y,r.total,color="#E5E9ED",height=.4); ax.barh(y,r.accepted,color=color,height=.4)
        ax.text(r.total*.08,y+.32,f"{int(r.accepted)}/{int(r.total)} ({r.rate_pct:.2f}%)",fontsize=8.0,color=color)
    ax.set_yticks([0,1],["Node-only","Full"]); ax.set_xlim(0,10000); ax.set_ylim(-.5,1.6); ax.set_xlabel("Evaluation records"); clean(ax)
    ax=axis_mm(fig,124,24,51,71)
    data=read("holdout_confidence_archive.csv"); labels=["Angular resolution","Normalized angular\nresolution","Crossing angle","Crossing angle, p05",
            "Node-edge clearance","Clearance, p05","Neighborhood\npreservation (k=5)"]
    metric=["angular_resolution_deg","angular_resolution_ratio","crossing_angle_deg","crossing_angle_p05_deg","node_edge_clearance","node_edge_clearance_p05","neighborhood_preservation_k5"]
    for comp,color,offset,marker in [("full_nodrop",BLUE,.14,"o"),("safe_full_vs_safe_node",GREEN,-.14,"s")]:
        for i,m in enumerate(metric):
            r=data[(data.dataset=="ALL")&(data.comparison==comp)&(data.metric==m)].iloc[0]
            ci(ax,r.mean_gain_pct,r.ci_low_pct,r.ci_high_pct,6-i+offset,color,marker)
    ax.axvline(0,color=GRAY,lw=.7); ax.set_yticks(range(7),labels[::-1]); ax.set_ylim(-.6,6.6)
    ax.set_xlabel("Improvement (%)"); clean(ax)
    fig.legend(handles=[Line2D([],[],marker="o",color=BLUE,lw=1,label="Full-safe vs Matching Backbone"),
                        Line2D([],[],marker="s",color=GREEN,lw=1,label="Full-safe vs Node-safe")],
               frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.51,.015))
    save(fig,"safe_holdout_forest")


def mechanism():
    fig=figure(127)
    mech=read("mechanism_confidence_archive.csv"); obj=read("objective_confidence_archive.csv")
    groups=[(mech,["full_raw_vs_same_weight_gate_zero_raw","full_raw_vs_same_weight_gate_one_raw","full_raw_vs_top_down_only_raw","full_raw_vs_bottom_up_only_raw"],
             ["Gate-zero","Gate-one","Top-down only","Bottom-up only"],"(a) Same-weight mechanism controls",115,72),
            (obj,["no_caid_target_nodrop","no_macpsg_nodrop","no_caid_macpsg_nodrop"],
             ["Remove CAID target","Remove MA-CPSG","Remove both"],"(b) Objective removals",57,17)]
    for frame,keys,labels,title,top,bottom in groups:
        text_mm(fig,4,top+6,title,9.1,weight="bold")
        for col,(metric,color) in enumerate([("graph_stress",BLUE),("community_sep_comp_ratio",GREEN)]):
            ax=axis_mm(fig,41+col*74,bottom,58,top-bottom)
            for i,key in enumerate(keys):
                if frame is mech:
                    r=frame[(frame.dataset=="ALL")&(frame.metric==metric)&(frame.comparison==key)].iloc[0]
                    low,high=r.ci_low_pct,r.ci_high_pct
                else:
                    r=frame[(frame.dataset=="ALL")&(frame.metric==metric)&(frame.reference_variant==key)].iloc[0]
                    low,high=r.hierarchical_bootstrap_ci_low_pct,r.hierarchical_bootstrap_ci_high_pct
                ci(ax,r.mean_gain_pct,low,high,len(keys)-1-i,color)
            ax.axvline(0,color=GRAY,lw=.7); ax.set_yticks(range(len(keys)),labels[::-1] if col==0 else [""]*len(keys))
            ax.set_ylim(-.5,len(keys)-.5); clean(ax)
            ax.set_xlabel("Stress gain (%)" if col==0 else "Sep/Comp gain (%)")
    save(fig,"hctf_mechanism_ablation")


def timing():
    fig=figure(111); summary=read("timing_summary.csv"); obs=read("timing_observations.csv")
    if (summary[["num_nodes","total_ms_median","total_ms_min","total_ms_max"]] <= 0).any().any():
        raise ValueError("Log timing axes require strictly positive graph sizes and total times.")
    names={"baseline":"Backbone","node_only_nodrop":"Node-only","full_nodrop":"Full"}; colors={"baseline":GRAY,"node_only_nodrop":GREEN,"full_nodrop":BLUE}
    text_mm(fig,4,105,"(a) Complete workflow",9.1,weight="bold")
    text_mm(fig,75,105,"(b) Stages at 128 nodes",9.1,weight="bold")
    text_mm(fig,141,105,"(c) Forward logs",9.1,weight="bold")
    ax=axis_mm(fig,19,31,47,65)
    for variant in names:
        d=summary[summary.variant==variant].sort_values("num_nodes")
        ax.plot(d.num_nodes,d.total_ms_median,"o-",ms=3,lw=1,color=colors[variant],label=names[variant])
        ax.fill_between(d.num_nodes.to_numpy(),d.total_ms_min.to_numpy(),d.total_ms_max.to_numpy(),color=colors[variant],alpha=.1)
    ax.set_xscale("log",base=2); ax.set_yscale("log")
    ax.set_xticks([16,32,64,128,256,512],["16","32","64","128","256","512"])
    ax.set_yticks([100,1000,5000],["100","1000","5000"])
    ax.set_xlabel("Nodes"); ax.set_ylabel("End-to-end ms/graph"); ax.legend(frameon=False,fontsize=8.0,loc="upper left"); clean(ax,"both")
    ax=axis_mm(fig,83,31,40,65); d=summary[summary.num_nodes==128].set_index("variant")
    stages=[("preprocess_ms_median","Preprocess","#C5CED6"),("baseline_forward_ms_median","Extra backbone",GRAY),
            ("proposal_forward_ms_median","Primary forward",BLUE),("audit_ms_median","Audit",ORANGE)]
    x=np.arange(3)
    for j,(metric,label,color) in enumerate(stages):
        ax.bar(x+(j-1.5)*.18,[d.loc[v,metric] for v in names],width=.18,color=color,label=label)
    ax.set_xticks(x,list(names.values()),rotation=22,ha="right",rotation_mode="anchor"); ax.set_ylabel("Median ms/graph"); clean(ax,"y")
    handles,labels=ax.get_legend_handles_labels(); fig.legend(handles,labels,ncol=2,frameon=False,loc="lower center",bbox_to_anchor=(.53,.01))
    ax=axis_mm(fig,145,31,31,65); attrib=read("model_only_attributed_archive.csv")
    ax.bar(range(3),attrib.ms,color=[GRAY,GREEN,BLUE],width=.6)
    ax.set_xticks(range(3),list(names.values()),rotation=22,ha="right",rotation_mode="anchor"); ax.set_ylim(0,38)
    for i,v in enumerate(attrib.ms): ax.text(i,v+1,f"{v:.1f}",ha="center",fontsize=8.0)
    ax.set_ylabel("Attributed ms/graph"); clean(ax,"y")
    save(fig,"end_to_end_scaling")


def all17():
    fig=figure(111); cov=read("all17_coverage_runtime.csv").set_index("dataset"); effects=read("all17_effects.csv")
    text_mm(fig,4,105,"(a) All-17 acceptance",9.0,weight="bold")
    text_mm(fig,80,105,"(b) Audited metrics",9.0,weight="bold")
    text_mm(fig,150,105,"(c) Audit cost",9.0,weight="bold")
    ax=axis_mm(fig,23,29,34,65)
    for y,ds,color in [(1,"enzymes",BLUE),(0,"proteins",GREEN)]:
        r=cov.loc[ds]; ax.barh(y,r.total,color="#E5E9ED",height=.42); ax.barh(y,r.accepted,color=color,height=.42)
        ax.text(2,y+.30,f"{int(r.accepted)}/{int(r.total)} ({r.rate_pct:.2f}%)",fontsize=8.0,color=color)
    ax.set_yticks([0,1],["PROTEINS","ENZYMES"]); ax.set_xlim(0,115); ax.set_ylim(-.5,1.65); ax.set_xlabel("Test graphs"); clean(ax)
    ax=axis_mm(fig,101,29,43,65)
    keys=["batch_stress","edge_crossing","edge_crossing_norm","edge_length_cv","node_occlusion_rate","community_silhouette"]
    labels=["Stress (pair sum)","Crossings","Normalized\ncrossings","Edge-length CV","Occlusion","Silhouette"]
    for ds,color,offset,marker in [("enzymes",BLUE,.13,"o"),("proteins",GREEN,-.13,"s")]:
        for i,key in enumerate(keys):
            r=effects[(effects.dataset==ds)&(effects.metric==key)].iloc[0]
            if pd.isna(r.low): ax.plot(r["mean"],5-i+offset,marker,color=color,ms=3.8)
            else: ci(ax,r["mean"],r.low,r.high,5-i+offset,color,marker)
    ax.axvline(0,color=GRAY,lw=.7); ax.set_yticks(range(6),labels[::-1]); ax.set_ylim(-.6,5.6); ax.set_xlabel("Metric gain (%)"); clean(ax)
    ax=axis_mm(fig,158,29,18,65)
    values=[cov.loc[d,"audit_seconds"] for d in ("enzymes","proteins")]
    ax.bar([0,1],values,color=[BLUE,GREEN],width=.6); ax.set_xticks([0,1],["ENZ","PRO"],rotation=25,ha="right",rotation_mode="anchor")
    for i,v in enumerate(values): ax.text(i,v+.5,f"{v:.2f}",ha="center",fontsize=8.0)
    ax.set_ylim(0,max(values)*1.28); ax.set_xlabel("s/graph"); clean(ax,"y")
    fig.legend(handles=[Line2D([],[],marker="o",linestyle="none",color=BLUE,label="ENZYMES: point estimates"),
                        Line2D([],[],marker="s",linestyle="none",color=GREEN,label="PROTEINS: paired 95% CI")],
               frameon=False,loc="lower center",bbox_to_anchor=(.53,.015),ncol=2)
    save(fig,"all17_deployment_analysis")


def tradeoff():
    fig=figure(112); data=read("stress_community_pairs.csv")
    limits={col:(float(data[col].min()),float(data[col].max())) for col in ["stress_gain_pct","community_gain_pct"]}
    for j,method in enumerate(["Raw Full","Full-safe"]):
        text_mm(fig,4+j*91,106,f"({chr(97+j)}) {method} vs Matching Backbone",9.0,weight="bold")
        ax=axis_mm(fig,19+j*88,31,66,65)
        part=data[data.method==method]
        for ds,color in zip(DS,PALETTE):
            d=part[part.dataset==ds]
            ax.scatter(d.stress_gain_pct,d.community_gain_pct,s=6,alpha=.42,color=color,edgecolors="none",label=NAMES[ds],zorder=2)
        ax.set_xscale("symlog",linthresh=1); ax.set_yscale("symlog",linthresh=1)
        ax.axvline(0,color=GRAY,lw=.7,zorder=1); ax.axhline(0,color=GRAY,lw=.7,zorder=1)
        for axis,col in [("x","stress_gain_pct"),("y","community_gain_pct")]:
            lo,hi=limits[col]; margin=.05*max(abs(lo),abs(hi)); lo-=margin; hi+=margin
            ticks=[v for v in [-10000,-1000,-100,-10,-1,0,1,10,100,1000,10000] if lo<=v<=hi]
            if axis=="x": ax.set_xlim(lo,hi); ax.set_xticks(ticks,[str(v) for v in ticks])
            else: ax.set_ylim(lo,hi); ax.set_yticks(ticks,[str(v) for v in ticks])
        ax.set_xlabel("Stress improvement (%)")
        ax.set_ylabel("Sep/Comp improvement (%)" if j==0 else "")
        ax.text(.03,.97,f"{len(part):,} records",transform=ax.transAxes,ha="left",va="top",fontsize=8.0)
        if method=="Full-safe":
            fallback=int((read("safe_acceptance.csv").query("variant == 'full_nodrop'").iloc[0]["total"]-
                          read("safe_acceptance.csv").query("variant == 'full_nodrop'").iloc[0]["accepted"]))
            ax.text(.03,.91,f"{fallback:,} exact fallbacks at zero",transform=ax.transAxes,
                    ha="left",va="top",fontsize=8.0,color=GRAY)
        clean(ax,"both")
    handles=[Line2D([],[],marker="o",linestyle="none",color=c,ms=4,label=NAMES[d]) for d,c in zip(DS,PALETTE)]
    fig.legend(handles=handles,ncol=4,loc="lower center",bbox_to_anchor=(.53,.005),frameon=False,columnspacing=1.2)
    save(fig,"stress_community_tradeoff")


def main():
    qualitative(); community_readability(); paired_structure(); safe_holdout(); mechanism(); timing(); all17(); tradeoff()


if __name__=="__main__":
    main()
