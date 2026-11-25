"""Code-generated community-objective schematic; no experimental layout claim.

Run: python draw_fig02_unified.py
The accompanying JSON preserves all original conceptual geometry and topology.
"""
from pathlib import Path
from io import BytesIO
import base64
import json
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from fontTools import subset
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
STEM='Fig04_Community_Objectives'
WIDTH_MM=180.0
HEIGHT_MM=82.0
DPI=600
FONT='Arial'
P={'white':'#FFFFFF','blue':'#E8F2FB','orange':'#F1C99F','cyan':'#BBDCE5',
   'purple':'#E7DDF1','ink':'#20272D','border':'#789DBA','edge':'#86929B','accent':'#947EA9'}
COMMUNITY=['#6399CC','#E4A05A','#89B25A']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8.4,'mathtext.fontset':'stix','mathtext.rm':FONT,
    'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})
TEXTS=[];BOXES={};ROUTES=[];CHECKS={}

def label(ax,x,y,value,size=8.4,weight='normal',ha='center',name=None,owner=None,color=None):
    obj=ax.text(x,y,value,fontsize=size,fontweight=weight,ha=ha,va='center',
                linespacing=1.1,color=color or P['ink'],zorder=10)
    obj.set_gid(name or f'label_{len(TEXTS)}');TEXTS.append((obj,owner));return obj

def box(ax,name,x,y,w,h,fill='white',dashed=False,text=None,lw=.65):
    patch=FancyBboxPatch((x,y),w,h,
        boxstyle=f'round,pad=0,rounding_size={1.5 if dashed else .85}',
        facecolor=P[fill],edgecolor=P['border'] if dashed or fill=='white' else P['ink'],
        linewidth=lw,linestyle=(0,(4,2.7)) if dashed else '-',zorder=1 if dashed else 2)
    patch.set_gid(name);ax.add_patch(patch);BOXES[name]=[x,y,w,h]
    if text:label(ax,x+w/2,y+h/2,text,name=name+'_text',owner=name)

def line(ax,name,points,color=None,lw=.65,dashed=False,zorder=3):
    a=np.asarray(points,float)
    ax.plot(a[:,0],a[:,1],color=color or P['edge'],lw=lw,
            linestyle=(0,(2.4,1.7)) if dashed else '-',solid_capstyle='round',zorder=zorder,gid=name)

def arrow(ax,name,points):
    points=np.asarray(points,float)
    patch=FancyArrowPatch(path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)),
        arrowstyle='-|>',mutation_scale=6.0,linewidth=.65,color=P['ink'],shrinkA=0,shrinkB=0,
        joinstyle='miter',capstyle='butt',zorder=4)
    patch.set_gid(name);ax.add_patch(patch);ROUTES.append({'name':name,'points':points.tolist()})

def node(ax,name,xy,community=0,radius=1.10,boundary=False):
    patch=Circle(xy,radius,facecolor=COMMUNITY[community],edgecolor=P['ink'],lw=.38,zorder=6)
    patch.set_gid(name);ax.add_patch(patch)
    if boundary:
        ring=Circle(xy,radius+.85,facecolor='none',edgecolor=COMMUNITY[1],lw=.72,zorder=5)
        ring.set_gid(name+'_boundary');ax.add_patch(ring)

def prototype(ax,name,xy,community=0,radius=1.25):
    patch=Circle(xy,radius,facecolor=P['white'],edgecolor=COMMUNITY[community],lw=.8,zorder=7)
    patch.set_gid(name);ax.add_patch(patch)
    for suffix,offsets in [('h',[[-.60,0],[.60,0]]),('v',[[0,-.60],[0,.60]])]:
        line(ax,name+'_'+suffix,np.asarray(xy)+np.asarray(offsets),color=COMMUNITY[community],lw=.65,zorder=8)

def cloud(ax,name,center,radius):
    patch=Circle(center,radius,facecolor=P['cyan'],edgecolor=P['border'],lw=.5,alpha=.42,zorder=3)
    patch.set_gid(name);ax.add_patch(patch)

def transformed(points,old_center,new_center,scale=1.):
    return (np.asarray(points,float)-np.asarray(old_center))*scale+np.asarray(new_center)

def draw_caid(ax,data):
    box(ax,'caid_graph_field',7,44,70,22,fill='white',lw=.45)
    points=np.asarray(data['CAID_POINTS'],float);labels=np.asarray(data['CAID_LABELS'],int)
    edges=np.asarray(data['CAID_EDGES'],int)
    xy=transformed(points,[42,54],[40.8,56],.9)
    adjacency=np.zeros((len(points),len(points)),int)
    for u,v in edges:adjacency[u,v]=adjacency[v,u]=1
    boundary=(adjacency*(labels[:,None]!=labels[None,:])).sum(1)/(adjacency.sum(1)+1e-6)
    for k,original_center in enumerate([[18,54],[42,54],[67,54]]):
        center=transformed([original_center],[42,54],[40.8,56],.9)[0]
        cloud(ax,'caid_community_'+str(k),center,8.19)
        label(ax,center[0],69,'ABC'[k],weight='bold',color=COMMUNITY[k],name='community_'+str(k))
    for u,v in edges:line(ax,f'caid_edge_{u}_{v}',xy[[u,v]],lw=.62)
    for n,position in enumerate(xy):node(ax,'caid_node_'+str(n),position,int(labels[n]),boundary=boundary[n]>0)
    for center,values,title,dashed in [(18,(0,0),'Same',False),(41,(0,1),'Adjacent',False),(64,(0,2),'Non-adjacent',True)]:
        endpoints=[[center-3.5,39.3],[center+3.5,39.3]]
        line(ax,'relation_'+title,endpoints,lw=.65,dashed=dashed)
        for j,point in enumerate(endpoints):node(ax,'relation_'+title+'_'+str(j),point,values[j],radius=.78)
        label(ax,center,35.2,title,name='relation_label_'+title)
    node(ax,'boundary_key',[39,28],0,radius=.85,boundary=True)
    label(ax,55,28,'Weighting',name='weighting_label')
    box(ax,'original_target',9,11,25,9,fill='cyan',text='Original\nTarget')
    box(ax,'adjusted_target',45,11,28,9,fill='orange',text='Training\nTarget')
    arrow(ax,'target_adjustment',[[34,15.5],[45,15.5]])
    arrow(ax,'weighting_to_target',[[59,25],[59,20]])
    CHECKS['caid']={'nodes':list(range(len(xy))),'edges':edges.tolist(),'communities':labels.tolist(),
        'boundary_nodes':np.flatnonzero(boundary>0).tolist(),'display_positions_mm':xy.tolist(),
        'display_scale':.9,'original_center':[42,54],'display_center':[40.8,56]}

def draw_geometry(ax,data):
    for name,x,title in [('compactness',89,'Compactness'),('silhouette',118.5,'Silhouette'),('separation',148,'Separation')]:
        box(ax,name+'_field',x,33,26,29,fill='white',lw=.45)
        label(ax,x+13,66.5,title,name=name+'_title')
    pts=transformed(data['PROTO_POINTS'],[103,48.5],[102,47.8])
    proto_center=pts.mean(0)
    cloud(ax,'prototype_cloud',[102,47.8],9.5)
    for n,point in enumerate(pts):node(ax,'prototype_node_'+str(n),point,radius=1.05)
    line(ax,'node_own_prototype',[pts[-1],proto_center],dashed=True,color=P['accent'])
    prototype(ax,'own_center',proto_center)
    own=transformed(data['SIL_OWN_POINTS'],[139,49],[131.5,47.8],.82)
    other=transformed(data['SIL_OTHER_POINTS'],[139,49],[131.5,47.8],.82)
    own_center=own.mean(0);other_center=other.mean(0);focal=own[-1]
    for name,source_center in [('own',[132,49]),('other',[146,49])]:
        cloud(ax,'silhouette_'+name+'_cloud',transformed([source_center],[139,49],[131.5,47.8],.82)[0],5.002)
    for k,points in enumerate([own,other]):
        for n,point in enumerate(points):node(ax,f'silhouette_node_{k}_{n}',point,k,radius=.92)
    for name,target in [('own',own_center),('other',other_center)]:
        line(ax,'focal_to_'+name,[focal,target],dashed=True,color=P['accent'],lw=.7,zorder=6)
    prototype(ax,'silhouette_own_center',own_center,0,radius=1.03)
    prototype(ax,'silhouette_other_center',other_center,1,radius=1.03)
    focus=Circle(focal,1.42,facecolor='none',edgecolor=P['accent'],lw=.65,zorder=7)
    focus.set_gid('focal_node_ring');ax.add_patch(focus)
    prototypes=[]
    for k,center in enumerate([[153,47.8],[161,48.76],[169,47.8]]):
        cloud(ax,'separation_cloud_'+str(k),center,3.2)
        nodes=np.asarray(data['SEP_OFFSETS'])*.8+center
        for n,point in enumerate(nodes):node(ax,f'separation_node_{k}_{n}',point,k,radius=.48)
        prototypes.append(nodes.mean(0))
        prototype(ax,'separation_center_'+str(k),nodes.mean(0),k,radius=.82)
    line(ax,'adjacent_center_relations',prototypes,color=P['edge'],lw=.65,zorder=5)
    far=transformed([[158,55],[161,58],[168,59],[175,58],[178,55]],[168,48],[161,47.8],.8)
    line(ax,'far_relation',far,dashed=True,color=P['accent'],lw=.65)
    box(ax,'community_loss',96,12,69,9,fill='orange')
    label(ax,130.5,16.5,r'$\mathcal{L}_{\mathrm{comm}}$',size=10.,name='community_loss_value',owner='community_loss')
    for name,x in [('compactness',102),('silhouette',131.5),('separation',161)]:arrow(ax,name+'_to_loss',[[x,33],[x,21]])
    CHECKS['geometry']={'prototype_points_mm':pts.tolist(),'prototype_center_mm':proto_center.tolist(),
        'silhouette_own_points_mm':own.tolist(),'silhouette_other_points_mm':other.tolist(),
        'own_prototype_mm':own_center.tolist(),'other_prototype_mm':other_center.tolist(),
        'focal_node_mm':focal.tolist(),'separation_centers_mm':[p.tolist() for p in prototypes]}

def geometry(fig,ax):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[];outside=[];overflow=[]
    for obj,owner in TEXTS:
        bb=obj.get_window_extent(renderer).transformed(ax.transData.inverted())
        row={'name':obj.get_gid(),'text':obj.get_text(),'bounds_mm':list(bb.extents),'owner':owner};bounds.append(row)
        if bb.x0<2.5 or bb.x1>WIDTH_MM-2.5 or bb.y0<2.5 or bb.y1>HEIGHT_MM-2.5:outside.append(row)
        if owner:
            x,y,w,h=BOXES[owner]
            if bb.x0<x+.1 or bb.x1>x+w-.1 or bb.y0<y+.1 or bb.y1>y+h-.1:overflow.append(row)
    collisions=[];intersections=[]
    for i,left in enumerate(bounds):
        for right in bounds[i+1:]:
            a=left['bounds_mm'];b=right['bounds_mm']
            if min(a[2],b[2])-max(a[0],b[0])>.07 and min(a[3],b[3])-max(a[1],b[1])>.07:collisions.append([left['name'],right['name']])
    for route in ROUTES:
        for text in bounds:
            x0,y0,x1,y1=text['bounds_mm']
            if MplPath(route['points']).intersects_bbox(Bbox.from_extents(x0-.07,y0-.07,x1+.07,y1+.07),filled=False):
                intersections.append([route['name'],text['name']])
    return {'text_bounds':bounds,'text_outside_safe_canvas':outside,'text_owner_overflow':overflow,
        'text_collisions':collisions,'connector_text_intersections':intersections,'boxes':BOXES,'routes':ROUTES,
        'scientific_data':CHECKS,'physical_size_mm':[WIDTH_MM,HEIGHT_MM]}

def embed_svg_fonts(path):
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    tree=ET.parse(path);root=tree.getroot()
    chars=''.join(''.join(n.itertext()) for n in root.findall(f'.//{{{ns}}}text'));rules=[]
    families=[(FONT,'normal','normal'),(FONT,'bold','normal'),(FONT,'normal','italic'),
              ('STIXGeneral','normal','normal'),('STIXGeneral','normal','italic'),
              ('STIXNonUnicode','normal','italic')]
    for family,weight,style in families:
        options=subset.Options();options.drop_tables+=['FFTM','meta']
        font=subset.load_font(font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style)),options)
        sub=subset.Subsetter(options=options);sub.populate(text=chars);sub.subset(font)
        stream=BytesIO();font.save(stream)
        rules.append(f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:{style};src:url(data:font/ttf;base64,{base64.b64encode(stream.getvalue()).decode()}) format('truetype');}}")
    style=ET.SubElement(root.find(f'{{{ns}}}defs'),f'{{{ns}}}style',{'type':'text/css'});style.text='\n'.join(rules)
    tree.write(path,encoding='utf-8',xml_declaration=True)

def render():
    data=json.loads((OUT/'COMMUNITY_SOURCE_DATA.json').read_text(encoding='utf-8'))
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=300)
    ax=fig.add_axes([0,0,1,1],xlim=(0,WIDTH_MM),ylim=(0,HEIGHT_MM),aspect='equal');ax.axis('off')
    box(ax,'caid_panel',3,3,78,76,fill='blue',dashed=True)
    box(ax,'macpsg_panel',84,3,93,76,fill='blue',dashed=True)
    for x,letter,title in [(6,'a','CAID'),(87,'b','MA-CPSG')]:
        label(ax,x,75,letter,size=8.6,weight='bold',ha='left')
        label(ax,x+4.5,75,title,size=8.4,weight='bold',ha='left')
    draw_caid(ax,data);draw_geometry(ax,data)
    audit=geometry(fig,ax)
    for key in ['text_outside_safe_canvas','text_owner_overflow','text_collisions','connector_text_intersections']:assert not audit[key],(key,audit[key])
    fig.savefig(OUT/f'{STEM}.pdf')
    fig.savefig(OUT/f'{STEM}.svg');embed_svg_fonts(OUT/f'{STEM}.svg')
    fig.savefig(OUT/f'{STEM}.png',dpi=DPI)
    with Image.open(OUT/f'{STEM}.png') as im:im.convert('RGB').save(OUT/f'{STEM}.tiff',dpi=(600,600),compression='raw')
    fig.savefig(OUT/'QA_180mm.png',dpi=300);plt.close(fig)
    (OUT/'GEOMETRY_QA.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'status':'SOURCE_RENDER_AND_GEOMETRY_PASS','text_elements':len(TEXTS),'size_mm':[WIDTH_MM,HEIGHT_MM]}))

if __name__=='__main__':render()
