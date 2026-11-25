# -*- coding: utf-8 -*-
"""Python-only editable SafeBlend mechanism figure, 180 x 118 mm.

Style inherited from Figure 1 V4. Real full-graph initial/X0/X1/X* coordinates
are read from the audited source bundle; no coordinate optimisation occurs.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from fontTools import subset
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
STEM = 'Fig03_SafeBlend_Protocol'
WIDTH_MM=180.0
HEIGHT_MM=118.0
FONT='Arial'
P={'white':'#FFFFFF','blue':'#E8F2FB','orange':'#F1C99F','cyan':'#BBDCE5',
   'purple':'#E7DDF1','ink':'#20272D','border':'#789DBA','edge':'#86929B'}
COMMUNITY_COLORS=['#6399CC','#E4A05A','#89B25A','#365E86','#4D7B48']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
    'font.size':7.,'mathtext.fontset':'custom','mathtext.rm':FONT,
    'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})
TEXTS=[];BOXES={};ROUTES=[];GLYPHS=[]

def text(ax,x,y,value,size=7.,weight='normal',ha='center',name=None,owner=None):
    t=ax.text(x,y,value,fontsize=size,fontweight=weight,color=P['ink'],ha=ha,
              va='center',linespacing=1.1,zorder=10)
    t.set_gid(name or f'label_{len(TEXTS)}');TEXTS.append((t,owner));return t

def box(ax,name,x,y,w,h,value='',fill='orange',dashed=False,radius=.85,lw=.65,size=7.,weight='normal'):
    b=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=P[fill],edgecolor=P['border'] if dashed else P['ink'],linewidth=lw,
        linestyle=(0,(4,2.7)) if dashed else '-',zorder=1 if dashed else 5)
    b.set_gid(name);ax.add_patch(b);BOXES[name]=[x,y,w,h]
    if value:text(ax,x+w/2,y+h/2,value,size=size,weight=weight,name=name+'_text',owner=name)

def route(ax,name,points,arrow=True,lw=.65,zorder=3,draw_segments=None):
    points=np.array(points,dtype=float)
    segments=draw_segments or [points.tolist()]
    vertices=[];codes=[]
    for segment in segments:
        vertices.extend(segment);codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(segment)-1))
    p=FancyArrowPatch(path=MplPath(vertices,codes),
        arrowstyle='-|>' if arrow else '-',mutation_scale=6.,linewidth=lw,color=P['ink'],
        shrinkA=0,shrinkB=0,capstyle='butt',joinstyle='miter',zorder=zorder)
    p.set_gid(name);ax.add_patch(p)
    entry={'name':name,'points':points.tolist(),'arrow':arrow,'linewidth_pt':lw}
    if draw_segments is not None:entry['draw_segments']=draw_segments
    ROUTES.append(entry)

def junction(ax,x,y):ax.add_patch(Circle((x,y),.30,facecolor=P['ink'],edgecolor='none',zorder=4))

def network(ax,name,coords,edges,c,center):
    xy=coords*24.3+np.array(center)
    for u,v in edges:
        ax.plot(xy[[u,v],0],xy[[u,v],1],color=P['edge'],alpha=.65,lw=.26,zorder=6,
                solid_capstyle='round',gid=f'{name}_edge_{u}_{v}')
    for node,(x,y) in enumerate(xy):
        p=Circle((x,y),.33,facecolor=COMMUNITY_COLORS[int(c[node])],edgecolor='white',lw=.15,zorder=7)
        p.set_gid(f'{name}_node_{node}');ax.add_patch(p)
    GLYPHS.append({'name':name,'coordinates_mm':xy.tolist(),'nodes':list(range(len(xy))),
                   'edges':edges.tolist(),'communities':c.tolist(),'radius_mm':.33})

def embed_fonts(path):
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree=ET.parse(path);root=tree.getroot()
    for node in root.iter():
        if node.tag in (f'{{{ns}}}text',f'{{{ns}}}tspan') and 'style' in node.attrib:
            node.set('style',node.get('style').replace("'sans-serif'","'Arial'"))
    characters=''.join(''.join(n.itertext()) for n in root.findall(f'.//{{{ns}}}text'))
    rules=[]
    # widehat uses the mathematical extension font. Embed it as well so an
    # editable SVG does not substitute a visible Latin c for the hat glyph.
    for family,weight,style in [(FONT,'normal','normal'),(FONT,'bold','normal'),
                                (FONT,'normal','italic'),('cmex10','normal','normal')]:
        prop=font_manager.FontProperties(family=family,weight=weight,style=style)
        options=subset.Options();options.drop_tables+=['FFTM','meta']
        f=subset.load_font(font_manager.findfont(prop),options);s=subset.Subsetter(options=options)
        s.populate(text=characters);s.subset(f);buffer=BytesIO();f.save(buffer)
        rules.append(f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};src:url(data:font/ttf;base64,{base64.b64encode(buffer.getvalue()).decode()}) format('truetype');}}")
    node=ET.SubElement(root.find(f'{{{ns}}}defs'),f'{{{ns}}}style',{'type':'text/css','id':'embedded_editable_fonts'})
    node.text='\n'.join(rules);tree.write(path,encoding='utf-8',xml_declaration=True)

def draw(ax,data):
    # Shared translation and isotropic scale across every real graph view.
    keys=['input','x0','x1','aligned_x1','final']
    stacked=np.concatenate([data[k] for k in keys])
    center=(stacked.max(axis=0)+stacked.min(axis=0))/2
    span=float(np.ptp(stacked,axis=0).max());assert span>0
    coords={k:(data[k]-center)/span for k in keys}
    for name,x,y,w,h in [('generation',3,12.0,102,101.5),
                          ('audit',108,60,29,53.5),('selection',140,14.8,37,98.7)]:
        box(ax,name,x,y,w,h,fill='blue',dashed=True,radius=1.6)
    text(ax,6,110.4,'a',size=8.2,weight='bold',ha='left')
    text(ax,10.4,110.4,'Candidate Generation',size=7.5,weight='bold',ha='left')
    text(ax,110,110.4,'b',size=8.2,weight='bold',ha='left')
    text(ax,114.0,109.8,'Multi-criteria\nAudit',size=7.1,weight='bold',ha='left')
    text(ax,142,110.4,'c',size=8.2,weight='bold',ha='left')
    text(ax,146,110.4,'Select / Fallback',size=7.5,weight='bold',ha='left')

    views=[('input','input',6,59),('x0','x0',40,76),('x1','x1',73,76),
           ('aligned','aligned_x1',58,36.5),('final','final',145,25.5)]
    for name,key,x,y in views:
        box(ax,name+'_frame',x,y,27,27,fill='white')
        network(ax,name,coords[key],data['edges'],data['communities'],(x+13.5,y+13.5))
    text(ax,19.5,89.2,'Input Graph G',size=7.)
    # The lower 3 mm gap contains the alignment connectors; retain compact
    # centered variables above the frames rather than crowding those outputs.
    text(ax,53.5,106.0,r'$(X_0)$',size=10.,name='layout_x0_label',owner='generation')
    text(ax,86.5,106.0,r'$(X_1)$',size=10.,name='layout_x1_label',owner='generation')
    text(ax,68.85,33.5,'Aligned Proposal',size=7.,name='aligned_caption_name',owner='generation')
    text(ax,81.45,33.5,r'$\widehat X_1$',size=10.,name='aligned_caption_symbol',owner='generation')

    # One G bus feeds two independent inputs. Matching top ports and adjacent
    # labels remove the long-distance association of the old proposal label.
    route(ax,'graph_to_fork',[(33,85),(36,85),(36,108.5),(41,108.5)],arrow=False)
    junction(ax,41,108.5)
    route(ax,'graph_to_backbone',[(41,108.5),(41,103)])
    route(ax,'graph_to_proposal',[(41,108.5),(74,108.5),(74,103)])
    text(ax,42.5,110.4,'Backbone',size=7.,ha='left',name='backbone_path_label',owner='generation')
    text(ax,75.5,110.4,'HCTF-enhanced',size=7.,ha='left',name='hctf_path_label',owner='generation')
    box(ax,'alignment',59,66,23,7,'Procrustes\nAlignment',fill='cyan')
    route(ax,'x0_to_fork',[(53.5,76),(53.5,69.5)],arrow=False)
    junction(ax,53.5,69.5)
    route(ax,'x0_to_alignment',[(53.5,69.5),(59,69.5)])
    route(ax,'x1_to_alignment',[(86.5,76),(86.5,69.5),(82,69.5)])
    route(ax,'alignment_to_real_aligned',[(70.5,66),(70.5,63.5)])
    box(ax,'interpolation',40,23.5,62,7,fill='orange')
    text(ax,66.,27.0,'Candidate Interpolation',size=7.,owner='interpolation')
    text(ax,93.,27.0,r'$X(\alpha)$',size=10.,owner='interpolation')
    route(ax,'x0_to_interpolation',[(53.5,69.5),(53.5,30.5)])
    route(ax,'aligned_to_interpolation',[(85,50.0),(88,50.0),(88,30.5)])
    text(ax,46.4,39.5,'α = 0',size=7.)

    # Ordered categorical rail: equally spaced ticks are traversal slots, not
    # a numerical alpha axis. Labels are the exact six registered values.
    route(ax,'candidate_rail',[(15,17.0),(95,17.0)],arrow=False)
    tick_values=[.0625,.125,.25,.5,.75,1.]
    for i,(x,label) in enumerate(zip([17,32,47,62,77,92],['0.0625','0.125','0.25','0.5','0.75','1.0'])):
        route(ax,f'alpha_tick_{i+1}',[(x,16.35),(x,17.65)],arrow=False)
        text(ax,x,14.1,label,size=7.)
    text(ax,8.,17.0,r'$\alpha$',size=10.)
    # Two audit functions share a single instrument-like card. Its clipped
    # lower chamber is orange; the outer enclosure is drawn only once.
    box(ax,'audit_card',111,72.5,23,25.5,fill='cyan')
    outer=ax.patches[-1]
    lower=Rectangle((111,72.5),23,15.5,facecolor=P['orange'],edgecolor='none',zorder=5.1)
    lower.set_clip_path(outer);lower.set_gid('audit_lower_chamber');ax.add_patch(lower)
    outer.set_linewidth(0)
    outline=FancyBboxPatch((111,72.5),23,25.5,boxstyle='round,pad=0,rounding_size=.85',
        facecolor='none',edgecolor=P['ink'],linewidth=.65,zorder=5.2)
    outline.set_gid('audit_single_outline');ax.add_patch(outline)
    route(ax,'audit_divider',[(111,88),(134,88)],arrow=False,lw=.45,zorder=6)
    text(ax,122.5,95.3,'Metric Evaluation',size=7.2,owner='audit_card')
    text(ax,119.8,91.5,'Audit vs.',size=7.,name='audit_reference_name',owner='audit_card')
    text(ax,130.0,91.5,r'$X_0$',size=10.,name='audit_reference_symbol',owner='audit_card')
    text(ax,122.5,80.1,'Tolerance Check\n+ Minimum\nImprovement',size=7.1,owner='audit_card')
    route(ax,'candidate_to_metrics',[(102,27.0),(106.5,27.0),(106.5,93.5),(111,93.5)])
    route(ax,'audit_internal_flow',[(122.5,89.8),(122.5,87.0)],lw=.65,zorder=6)
    # PASS and FAIL are docked result markers on a genuine split output.
    route(ax,'audit_to_split',[(122.5,72.5),(122.5,68.5)],arrow=False)
    junction(ax,122.5,68.5)
    box(ax,'badge_pass',124,62.5,10,4,'PASS',fill='cyan',size=7.)
    box(ax,'badge_fail',110,62.5,10,4,'FAIL',fill='white',size=7.)
    route(ax,'split_to_pass',[(122.5,68.5),(129,68.5),(129,66.5)])
    route(ax,'split_to_fail',[(122.5,68.5),(115,68.5),(115,66.5)])

    # Display abstraction only: alpha_best and score_best remain actual
    # selector state, documented in Algorithm 2 and the audit, not repeated.
    box(ax,'incumbent_state',144.5,92,28,11,fill='purple')
    text(ax,152.5,99.8,'Incumbent',size=7.5,owner='incumbent_state')
    text(ax,166.2,99.8,r'$X_{\mathrm{best}}$',size=10.,owner='incumbent_state')
    text(ax,155.4,95.0,'Init:',size=7.,owner='incumbent_state')
    text(ax,162.0,95.0,r'$X_0$',size=10.,owner='incumbent_state')
    box(ax,'score',143.5,77,30,8,'Score + Tie-break',fill='orange',size=7.3)
    route(ax,'incumbent_to_score',[(158.5,92),(158.5,85)])
    route(ax,'audit_pass',[(134,64.5),(138.5,64.5),(138.5,81),(143.5,81)])
    # Update occupies the outside slot so its write-back is entirely separate
    # from the completion bus. Keep and FAIL never write the state register.
    box(ax,'keep',144.5,66,12.5,6.5,'Keep',fill='cyan',size=7.3)
    box(ax,'update',160,66,12.5,6.5,'Update',fill='purple',size=7.3)
    route(ax,'score_to_keep',[(150.75,77),(150.75,72.5)])
    route(ax,'score_to_update',[(166.25,77),(166.25,72.5)])
    text(ax,170.5,74.8,'wins',size=7.,name='winner_condition',owner='selection')
    text(ax,157.5,74.8,'otherwise',size=7.,name='keep_condition',owner='selection')
    route(ax,'fail_to_completion_bus',[(115,62.5),(115,61.5),(166.25,61.5)],arrow=False)
    route(ax,'keep_to_completion_bus',[(150.75,66),(150.75,61.5)],arrow=False)
    route(ax,'update_to_completion_bus',[(166.25,66),(166.25,61.5)],arrow=False)
    for x in [150.75,158.5,166.25]:junction(ax,x,61.5)
    route(ax,'update_state_feedback',[(172.5,69.25),(175.8,69.25),(175.8,97.5),(172.5,97.5)],lw=.45)

    # k is the processed nonzero-trial count implicit in the source for-loop.
    # All outcomes, including source 'continue', advance this traversal.
    controller=Circle((158.5,58.4),.75,facecolor=P['white'],edgecolor=P['ink'],linewidth=.65,zorder=5)
    controller.set_gid('loop_control_node');ax.add_patch(controller)
    BOXES['loop_control']=[157.75,57.65,1.5,1.5]
    route(ax,'completion_to_loop_control',[(158.5,61.5),(158.5,59.15)])
    # Conditions are inline edge labels. Genuine gaps in the vector strokes
    # preserve the background; no opaque mask or new decision box is used.
    route(ax,'next_alpha',[(157.75,58.4),(141.5,58.4),(141.5,13.2),(104.5,13.2),(104.5,21.5),(99,21.5),(99,23.5)],lw=.45,
          draw_segments=[[(157.75,58.4),(153,58.4)],[(146.4,58.4),(141.5,58.4),(141.5,13.2),(104.5,13.2),(104.5,21.5),(99,21.5),(99,23.5)]])
    text(ax,149.7,58.4,r'$k<6$',size=7.5,name='loop_not_done',owner='selection')
    text(ax,122.,16.0,'Next α',size=7.)
    route(ax,'six_complete_to_final',[(158.5,57.65),(158.5,52.5)],
          draw_segments=[[(158.5,57.65),(158.5,57.4)],[(158.5,54),(158.5,52.5)]])
    text(ax,158.5,55.8,r'$k=6$',size=7.5,name='loop_done',owner='selection')
    # Conditional exact fallback is stated once in the figure caption.
    text(ax,158.5,22.6,'Final Layout',size=7.2,name='final_caption_name',owner='selection')
    text(ax,158.5,18.3,r'$X^*=X_{\mathrm{best}}$',size=10.,name='final_caption_relation',owner='selection')

    # Two compact lines; no metric tiles and no arrows into online selection.
    route(ax,'footer_rule',[(3,11.0),(137,11.0)],arrow=False,lw=.45)
    text(ax,5,8.15,'Online Audit (10):',size=7.,ha='left')
    text(ax,5,4.25,'Post-selection (7):',size=7.,ha='left')
    text(ax,35,8.15,'Stress ×1 · Edge ×4 · Occlusion ×1 · Community ×4',size=7.,ha='left')
    text(ax,35,4.25,'Angular ×2 · Crossing ×2 · Clearance ×2 · Neighborhood ×1',size=7.,ha='left')
    return {'display_center':center.tolist(),'display_span':span,'common_mm_per_coordinate':24.3/span,
            'registered_alpha_values':tick_values,'tick_spacing':'ordered traversal slots, not linear numeric scale',
            'state_initialization':{'X_best':'stored X0','alpha_best':0,'score_best':0},
            'network_frame_mm':27.,'node_radius_mm':.33,
            'audit_compartments':{'metric_evaluation':[111,88,23,10],'feasibility':[111,72.5,23,15.5]},
            'result_markers':['badge_pass','badge_fail'],
             'state_feedback':{'writer':'update','route':'update_state_feedback','target':'incumbent_state',
                               'keep_and_fail_write':False},
             'candidate_completion':{'inputs':['update','keep','badge_fail'],'bus_y_mm':61.5,
                                     'route':'completion_to_loop_control'},
            'loop_counter':{'symbol':'k','meaning':'processed nonzero candidates including audit failures',
                            'next_when':'k < 6','final_when':'k = 6','center':[158.5,58.4],'radius_mm':.75},
            'inline_condition_labels':{'next_alpha':'loop_not_done','six_complete_to_final':'loop_done'}}

def geometry(fig,ax):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[];outside=[];overflow=[]
    for t,owner in TEXTS:
        b=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        row={'name':t.get_gid(),'text':t.get_text(),'bounds':list(b.extents),'owner':owner};bounds.append(row)
        if b.x0<2.5 or b.x1>177.5 or b.y0<1.5 or b.y1>HEIGHT_MM-1.5:outside.append(row)
        if owner:
            x,y,w,h=BOXES[owner]
            if b.x0<x+.1 or b.x1>x+w-.1 or b.y0<y+.1 or b.y1>y+h-.1:overflow.append(row)
    collisions=[]
    for i,a in enumerate(bounds):
        for b in bounds[i+1:]:
            x,y=a['bounds'],b['bounds']
            if min(x[2],y[2])-max(x[0],y[0])>.07 and min(x[3],y[3])-max(x[1],y[1])>.07:collisions.append([a['name'],b['name']])
    hits=[]
    for r in ROUTES:
        for t in bounds:
            x0,y0,x1,y1=t['bounds']
            if any(MplPath(s).intersects_bbox(Bbox.from_extents(x0-.07,y0-.07,x1+.07,y1+.07),filled=False)
                   for s in r.get('draw_segments',[r['points']])):hits.append([r['name'],t['name']])
    return {'text_bounds':bounds,'text_outside_safe_canvas':outside,'text_owner_overflow':overflow,
        'text_collisions':collisions,'connector_text_intersections':hits,'routes':ROUTES,'boxes':BOXES,'network_glyphs':GLYPHS}

def main():
    with np.load(OUT/'LAYOUT_SOURCE_DATA.npz') as src:data={k:src[k].copy() for k in src.files}
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=300)
    ax=fig.add_axes([0,0,1,1],xlim=(0,WIDTH_MM),ylim=(0,HEIGHT_MM),aspect='equal');ax.set_axis_off()
    display=draw(ax,data);audit=geometry(fig,ax);audit.update(display)
    fig.savefig(OUT/f'{STEM}.pdf')
    fig.savefig(OUT/f'{STEM}.svg')
    embed_fonts(OUT/f'{STEM}.svg')
    fig.savefig(OUT/f'{STEM}.png',dpi=600)
    with Image.open(OUT/f'{STEM}.png') as im:im.convert('RGB').save(OUT/f'{STEM}.tiff',dpi=(600,600),compression='raw')
    fig.savefig(OUT/'QA_Matplotlib_full.png',dpi=300)
    plt.close(fig)
    (OUT/'GEOMETRY_QA.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['text_outside_safe_canvas','text_owner_overflow','text_collisions','connector_text_intersections']},indent=2))

if __name__=='__main__':main()
