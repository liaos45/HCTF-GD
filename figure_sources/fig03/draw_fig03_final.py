# -*- coding: utf-8 -*-
"""Source-consistent Figure 3 (180 x 104 mm), integrated with the manuscript."""
from pathlib import Path
from io import BytesIO
import base64
import json
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle,FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from fontTools import subset
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
STEM = 'Fig02_HCTF_Structural_Block'
WIDTH_MM=180.;HEIGHT_MM=104.;FONT='Arial'
P={'white':'#FFFFFF','blue':'#E8F2FB','orange':'#F1C99F','cyan':'#BBDCE5',
   'purple':'#E7DDF1','ink':'#20272D','border':'#789DBA','operation':'#F7F7F7',
   'original':'#8A969F','reused':'#9EA8AF','f1':'#4D83A9','f2':'#3D9692','f3':'#947EA9'}
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7.,
    'mathtext.fontset':'custom','mathtext.rm':FONT,'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})
TEXTS=[];BOXES={};ROUTES=[];NETWORKS=[];DIRECTIONS=[]

def label(ax,x,y,s,size=7.,weight='normal',ha='center',name=None,owner=None):
    t=ax.text(x,y,s,fontsize=size,fontweight=weight,color=P['ink'],ha=ha,va='center',
              linespacing=1.1,zorder=10)
    t.set_gid(name or 'label_'+str(len(TEXTS)));TEXTS.append((t,owner));return t

def box(ax,name,x,y,w,h,fill='cyan',text=None,dashed=False,lw=.65):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={1.5 if dashed else .85}',
        facecolor=P[fill],edgecolor=P['border'] if dashed else P['ink'],lw=lw,
        linestyle=(0,(4,2.7)) if dashed else '-',zorder=1 if dashed else 4)
    p.set_gid(name);ax.add_patch(p);BOXES[name]=[x,y,w,h]
    if text:label(ax,x+w/2,y+h/2,text,owner=name,name=name+'_label')

def route(ax,name,points,arrow=True,color='ink',dashed=False,lw=.65,zorder=3):
    points=np.array(points,float)
    p=FancyArrowPatch(path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)),
        arrowstyle='-|>' if arrow else '-',mutation_scale=6.,shrinkA=0,shrinkB=0,
        linewidth=lw,color=P[color],linestyle=(0,(2.4,1.5)) if dashed else '-',
        capstyle='butt',joinstyle='miter',zorder=zorder)
    p.set_gid(name);ax.add_patch(p)
    ROUTES.append({'name':name,'points':points.tolist(),'arrow':arrow,'linewidth_pt':lw,'color':P[color],'dashed':dashed})

def junction(ax,x,y):ax.add_patch(Circle((x,y),.25,facecolor=P['ink'],edgecolor='none',zorder=5))

def graph_a(ax,name,cx,positions,edges,new_edges=None,color='original'):
    # One common transformation for all four views; no non-selected ghost edges.
    xy={int(k):(cx+4.6*v[0],79+7.2*v[1]) for k,v in positions.items()}
    card=FancyBboxPatch((cx-11.4,69.5),22.8,19.5,boxstyle='round,pad=0,rounding_size=.85',
                       facecolor='white',edgecolor='none',zorder=2)
    card.set_gid(name+'_field');ax.add_patch(card);BOXES[name+'_field']=[cx-11.4,69.5,22.8,19.5]
    new=set(tuple(e) for e in new_edges) if new_edges is not None else set(tuple(e) for e in edges)
    for u,v in edges:
        reused=new_edges is not None and (u,v) not in new
        a,b=xy[u],xy[v]
        line=ax.plot([a[0],b[0]],[a[1],b[1]],lw=.60 if reused else (.7 if name=='original' else .95),
            color=P['reused'] if reused else P[color],linestyle=(0,(2.2,1.3)) if reused else '-',
            solid_capstyle='round',zorder=3,gid=f'{name}_edge_{u}_{v}')[0]
    for n,(x,y) in xy.items():
        p=Circle((x,y),1.5,facecolor='white',edgecolor=P['ink'],lw=.55,zorder=6)
        p.set_gid(f'{name}_node_{n}');ax.add_patch(p)
        label(ax,x,y,str(n),size=7.,name=f'{name}_number_{n}')
    NETWORKS.append({'name':name,'positions_mm':xy,'nodes':list(xy),'edges':edges,
        'new_edges':list(new),'radius_mm':1.5,'node_fill':'#FFFFFF','node_edge':P['ink']})

def direction_edge(ax,u,v,positions,top_down):
    a=np.array(positions[u]);b=np.array(positions[v]);vector=b-a
    unit=vector/np.linalg.norm(vector);normal=np.array([-unit[1],unit[0]])
    shift=(.60 if top_down else -.60)*normal
    first=a+unit*2.05+shift;last=b-unit*2.05+shift
    if not top_down:first,last=last,first
    name=f'b_{"td" if top_down else "bu"}_{u}_{v}'
    p=FancyArrowPatch(tuple(first),tuple(last),arrowstyle='-|>',mutation_scale=6.3,
        shrinkA=0,shrinkB=0,color=P['f3' if top_down else 'f2'],lw=.8,
        linestyle='-' if top_down else (0,(2.4,1.4)),zorder=4)
    p.set_gid(name);ax.add_patch(p)
    DIRECTIONS.append({'name':name,'source':u if top_down else v,'target':v if top_down else u,
        'direction':'top_down' if top_down else 'bottom_up','forest':1,'start':first.tolist(),'end':last.tolist(),
        'dashed':not top_down,'color':P['f3' if top_down else 'f2']})

def residual(ax,name,x,y,w,h,title,symbol,fill):
    box(ax,name,x,y,w,h,fill=fill)
    label(ax,x+w/2,y+h-2.2,title,name=name+'_name',owner=name)
    label(ax,x+w/2,y+2.4,symbol,size=10.,name=name+'_symbol',owner=name)

def operation(ax,name,x,y,symbol):
    p=Circle((x,y),1.8,facecolor=P['operation'],edgecolor=P['ink'],lw=.65,zorder=6)
    p.set_gid(name);ax.add_patch(p);BOXES[name]=[x-1.8,y-1.8,3.6,3.6]
    label(ax,x,y,symbol,size=9.,name=name+'_symbol',owner=name)

def draw(data):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=300)
    ax=fig.add_axes([0,0,1,1],xlim=(0,180),ylim=(0,104),aspect='equal');ax.axis('off')
    box(ax,'panel_a',3,56,105,45,fill='blue',dashed=True)
    box(ax,'panel_b',111,56,66,45,fill='blue',dashed=True)
    box(ax,'panel_c',3,3,174,48,fill='blue',dashed=True)
    for x,y,letter,title in [(6,96.8,'a','Complementary Forest Construction'),
        (114,96.8,'b','Bidirectional Forest Aggregation'),(6,47.0,'c','Community-conditioned Residual Update')]:
        label(ax,x,y,letter,size=8.2,weight='bold',ha='left')
        label(ax,x+4.2,y,title,size=7.3,weight='bold',ha='left')
    centers=[16.5,42,67.5,93]
    graph_a(ax,'original',centers[0],data['positions'],data['original_edges'])
    label(ax,16.5,91.4,r'Original $G$',size=7.,name='original_caption')
    for k,f in enumerate(data['forests'],1):
        graph_a(ax,'forest_'+str(k),centers[k],data['positions'],f['edges'],f['new_edges'],'f'+str(k))
        label(ax,centers[k],91.4,r'$F_'+str(k)+'$',size=10.,name='forest_caption_'+str(k))
        label(ax,centers[k],67.1,'+'+str(f['new_count']),name='new_count_'+str(k))
    # The original edge set is the common source; no F1 -> F2 -> F3 chain.
    route(ax,'G_to_forest_bus',[(16.5,69.5),(16.5,64.8),(85,64.8)],arrow=False)
    for k,cx in enumerate(centers[1:],1):
        route(ax,'G_to_forest_'+str(k),[(cx-8,64.8),(cx-8,69.5)])
        if k<3:junction(ax,cx-8,64.8)
    route(ax,'new_edge_key',[(8,60.5),(13,60.5)],arrow=False,color='f1',lw=.95)
    label(ax,15,60.5,'New',ha='left')
    route(ax,'reused_edge_key',[(31,60.5),(36,60.5)],arrow=False,color='reused',dashed=True,lw=.6)
    label(ax,38,60.5,'Reused',ha='left')

    positions={6:(128,91),3:(116,81),4:(128,81),5:(140,81),1:(116,71)}
    for u,v in data['subtree_edges']:
        direction_edge(ax,u,v,positions,True);direction_edge(ax,u,v,positions,False)
    for n,(x,y) in positions.items():
        p=Circle((x,y),1.75,facecolor='white',edgecolor=P['ink'],lw=.55,zorder=6)
        p.set_gid(f'b_node_{n}');ax.add_patch(p);label(ax,x,y,str(n),name=f'b_number_{n}')
    # A collection bracket denotes the whole subtree, not just its last node.
    route(ax,'subtree_collection',[(143.6,93),(145,93),(145,69.5),(143.6,69.5)],arrow=False,lw=.45)
    BOXES['b_subtree_group']=[113.0,69.5,32,23.5]
    box(ax,'aggregation',148,78,25,9,fill='orange',text='Weighted\nAggregation')
    route(ax,'topdown_to_aggregation',[(145,84.3),(148,84.3)],color='f3')
    route(ax,'bottomup_to_aggregation',[(145,80.8),(148,80.8)],color='f2',dashed=True)
    residual(ax,'forest_context',148,64.5,25,9.5,'Forest Context',r'$t_i$','cyan')
    route(ax,'aggregation_to_context',[(160.5,78),(160.5,74)])
    route(ax,'topdown_key',[(113.5,60.5),(119,60.5)],arrow=False,color='f3',lw=.8)
    label(ax,120,60.5,'Top-down',ha='left')
    route(ax,'bottomup_key',[(143.8,60.5),(149.3,60.5)],arrow=False,color='f2',dashed=True,lw=.8)
    label(ax,150.3,60.5,'Bottom-up',ha='left')

    label(ax,8,28.25,r'$h_i$',size=10.,name='input_h')
    box(ax,'layernorm',17,25.25,17,6,fill='orange',text='LayerNorm')
    route(ax,'h_to_layernorm',[(11,28.25),(17,28.25)])
    junction(ax,13.5,28.25)
    route(ax,'normalized_trunk',[(34,28.25),(39,28.25)],arrow=False)
    junction(ax,37,28.25);junction(ax,39,28.25);junction(ax,39,33)
    label(ax,42.5,35.5,r'$\bar h_i$',size=10.,name='normalized_reference')
    residual(ax,'node_residual',57,35.5,27,8.5,'Node Residual',r'$u_i$','purple')
    route(ax,'normalized_to_node',[(39,28.25),(39,39.75),(57,39.75)])
    box(ax,'community_context',45,22,28,8.5,fill='cyan',text='Community\nContext')
    route(ax,'normalized_to_community',[(39,28.25),(45,28.25)])
    residual(ax,'structural_residual',83,22,27,8.5,'Structural Residual',r'$v_i$','purple')
    route(ax,'normalized_to_structural',[(39,33),(79,33),(79,28),(83,28)])
    route(ax,'community_to_structural',[(73,26.25),(83,26.25)])
    label(ax,100,35.7,r'$t_i$',size=10.,name='forest_reference')
    route(ax,'forest_to_structural',[(100,33.5),(100,30.5)])
    residual(ax,'graph_descriptor',45,8.5,28,10,'Graph Descriptor',r'$d_G$','cyan')
    route(ax,'normalized_to_descriptor',[(37,28.25),(37,20),(60,20),(60,18.5)])
    # A single external-input reference supplies both graph-local computations.
    label(ax,76.5,20.25,r'$G,c,b$',size=7.,name='graph_inputs')
    route(ax,'graph_input_source',[(80.7,20.25),(81.4,20.25)],arrow=False)
    junction(ax,81.4,20.25)
    route(ax,'graph_to_community',[(81.4,20.25),(81.4,23.5),(73,23.5)])
    route(ax,'graph_to_descriptor',[(81.4,20.25),(81.4,16.8),(73,16.8)])
    residual(ax,'adaptive_gate',82,8.5,28,10,'Adaptive Gate',r'$g_G$','orange')
    route(ax,'descriptor_to_gate',[(73,13.5),(82,13.5)])
    # Evaluation-time source-consistent fusion: h + s*a*(u + g*v).
    # s is a learned shared mixture times scale; a is node-wise modulation.
    for name,x,symbol in [('gate_multiply',121,'×'),('residual_add',134.5,'+'),
                          ('scale_multiply',149,'×'),('identity_add',163,'+')]:operation(ax,name,x,26.25,symbol)
    route(ax,'v_to_gate_multiply',[(110,26.25),(119.2,26.25)])
    route(ax,'g_to_gate_multiply',[(110,13.5),(121,13.5),(121,24.45)])
    route(ax,'gated_v_to_add',[(122.8,26.25),(132.7,26.25)])
    route(ax,'u_to_add',[(84,39.75),(134.5,39.75),(134.5,28.05)])
    route(ax,'sum_to_scale',[(136.3,26.25),(147.2,26.25)])
    box(ax,'scale',142,36,14,6,fill='orange')
    label(ax,149,39,r'$s\,a_i$',size=10.,name='scale_value',owner='scale')
    route(ax,'scale_to_multiply',[(149,36),(149,28.05)])
    route(ax,'scaled_to_identity',[(150.8,26.25),(161.2,26.25)])
    route(ax,'identity_skip',[(13.5,28.25),(13.5,6),(163,6),(163,24.45)])
    box(ax,'output',169,23.25,6.5,6,fill='purple')
    label(ax,172.25,26.25,r"$h'_i$",size=10.,name='output_h',owner='output')
    route(ax,'identity_to_output',[(164.8,26.25),(169,26.25)])
    return fig,ax

def embed_fonts(path):
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree=ET.parse(path);root=tree.getroot()
    for n in root.iter():
        if n.tag in (f'{{{ns}}}text',f'{{{ns}}}tspan') and 'style' in n.attrib:n.set('style',n.get('style').replace("'sans-serif'","'Arial'"))
    chars=''.join(''.join(n.itertext()) for n in root.findall(f'.//{{{ns}}}text'));rules=[]
    for weight,style in [('normal','normal'),('bold','normal'),('normal','italic')]:
        options=subset.Options();options.drop_tables+=['FFTM','meta']
        f=subset.load_font(font_manager.findfont(font_manager.FontProperties(family=FONT,weight=weight,style=style)),options)
        s=subset.Subsetter(options=options);s.populate(text=chars);s.subset(f);b=BytesIO();f.save(b)
        rules.append(f"@font-face{{font-family:'Arial';font-weight:{weight};font-style:{style};src:url(data:font/ttf;base64,{base64.b64encode(b.getvalue()).decode()}) format('truetype');}}")
    style=ET.SubElement(root.find(f'{{{ns}}}defs'),f'{{{ns}}}style',{'type':'text/css'});style.text='\n'.join(rules)
    tree.write(path,encoding='utf-8',xml_declaration=True)

def geometry(fig,ax):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[];outside=[];overflow=[]
    for t,owner in TEXTS:
        b=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        row={'name':t.get_gid(),'text':t.get_text(),'bounds':list(b.extents),'owner':owner};bounds.append(row)
        if b.x0<2.5 or b.x1>177.5 or b.y0<2.5 or b.y1>101.5:outside.append(row)
        if owner:
            x,y,w,h=BOXES[owner]
            if b.x0<x+.1 or b.x1>x+w-.1 or b.y0<y+.1 or b.y1>y+h-.1:overflow.append(row)
    collisions=[];crossings=[]
    for i,a in enumerate(bounds):
        for b in bounds[i+1:]:
            u,v=a['bounds'],b['bounds']
            if min(u[2],v[2])-max(u[0],v[0])>.07 and min(u[3],v[3])-max(u[1],v[1])>.07:collisions.append([a['name'],b['name']])
    for r in ROUTES:
        for t in bounds:
            x0,y0,x1,y1=t['bounds']
            if MplPath(r['points']).intersects_bbox(Bbox.from_extents(x0-.07,y0-.07,x1+.07,y1+.07),filled=False):crossings.append([r['name'],t['name']])
    return {'text_bounds':bounds,'text_outside_canvas':outside,'text_owner_overflow':overflow,'text_collisions':collisions,
        'connector_text_intersections':crossings,'boxes':BOXES,'routes':ROUTES,'networks':NETWORKS,'directions':DIRECTIONS}

def main():
    data=json.loads((OUT/'FOREST_AND_CODE_AUDIT_DATA.json').read_text(encoding='utf-8'))
    fig,ax=draw(data);g=geometry(fig,ax)
    fig.savefig(OUT/f'{STEM}.pdf');fig.savefig(OUT/f'{STEM}.svg');embed_fonts(OUT/f'{STEM}.svg')
    fig.savefig(OUT/f'{STEM}.png',dpi=600)
    with Image.open(OUT/f'{STEM}.png') as im:im.convert('RGB').save(OUT/f'{STEM}.tiff',dpi=(600,600),compression='raw')
    fig.savefig(OUT/'QA_Matplotlib_full.png',dpi=300);plt.close(fig)
    (OUT/'GEOMETRY_QA.json').write_text(json.dumps(g,indent=2),encoding='utf-8')
    print(json.dumps({k:g[k] for k in ['text_outside_canvas','text_owner_overflow','text_collisions','connector_text_intersections']},indent=2))

if __name__=='__main__':main()
