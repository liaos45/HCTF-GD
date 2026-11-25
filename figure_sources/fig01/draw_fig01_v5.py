"""Editable two-tier HCTF-GD architecture, 180 x 116 mm, Python only.

Contract: depict the gated configuration and its exact eval residual ordering.
The top views use one archived 48-node graph and unmodified saved coordinates.
Input G is a circular topological view. The lower forests remain a separate
nine-node constructor example. See STAGE_VIEW_AUDIT.json for provenance.
Run locally: python draw_fig01_v5.py
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
import networkx as nx
import numpy as np
from PIL import Image
from fontTools import subset

OUT = Path(__file__).resolve().parent
STEM = 'Fig01_Overall_Architecture'
WIDTH_MM = 180.0
HEIGHT_MM = 116.0
DPI = 600
FONT = 'DejaVu Sans'
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 7.0,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'DejaVu Sans', 'mathtext.it': 'DejaVu Sans:italic',
    'mathtext.bf': 'DejaVu Sans:bold',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.linewidth': 0.65,
})
P = {'white': '#FFFFFF', 'blue': '#E8F2FB', 'orange': '#F1C99F',
     'cyan': '#BBDCE5', 'purple': '#E7DDF1', 'op': '#F7F7F7',
     'ink': '#20272D', 'border': '#6F8EA9', 'edge': '#86929B'}
COMMUNITY_COLORS = ['#6399CC', '#E4A05A', '#89B25A', '#365E86', '#4D7B48', '#947EA9']
TEXTS, BOXES, ROUTES, GLYPHS = [], {}, [], []


def text(ax, x, y, value, size=7.0, weight='normal', ha='center', gid=None, owner=None):
    obj = ax.text(x, y, value, fontsize=size, fontweight=weight,
                  color=P['ink'], ha=ha, va='center', linespacing=1.12, zorder=10)
    obj.set_gid(gid or 'label_' + str(len(TEXTS)))
    TEXTS.append((obj, owner))
    return obj


def box(ax, name, x, y, w, h, value='', fill='orange', size=7.0,
        dashed=False, weight='normal', edge=None, radius=0.85, lw=0.65):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=P.get(fill, fill), edgecolor=edge or P['ink'],
                          linewidth=lw, linestyle=(0, (4, 2.7)) if dashed else '-',
                          zorder=1 if dashed else 5)
    patch.set_gid(name)
    ax.add_patch(patch)
    BOXES[name] = (x, y, w, h)
    if value:
        text(ax, x + w / 2, y + h / 2, value, size, weight=weight,
             gid=name + '_text', owner=name)
    return patch


def route(ax, name, points, arrow=True, dashed=False, lw=0.65, color=None, z=3):
    """Single editable path; orthogonal unless an enlargement guide is named."""
    points = np.asarray(points, dtype=float)
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>' if arrow else '-',
                            mutation_scale=6.0, linewidth=lw,
                            color=color or P['ink'],
                            linestyle=(0, (3.5, 2.6)) if dashed else '-',
                            shrinkA=0, shrinkB=0, capstyle='butt', joinstyle='miter', zorder=z)
    patch.set_gid(name)
    ax.add_patch(patch)
    ROUTES.append({'name': name, 'points_mm': points.tolist(), 'arrow': arrow,
                   'dashed': dashed})
    return patch


def junction(ax, x, y):
    ax.add_patch(Circle((x, y), 0.32, facecolor=P['ink'], edgecolor='none', zorder=4))


def op(ax, name, x, y, symbol):
    patch = Circle((x, y), 2.05, facecolor=P['op'], edgecolor=P['ink'],
                   linewidth=0.65, zorder=5)
    patch.set_gid(name)
    ax.add_patch(patch)
    text(ax, x, y + 0.02, symbol, size=10.0, gid=name + '_symbol')
    return patch


def normalized_positions(points):
    points = np.asarray(points, dtype=float)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    span = float(np.ptp(points, axis=0).max())
    assert span > 0 and np.isfinite(points).all()
    return (points - center) / span


def network(ax, name, positions, edges, communities, center, width, height,
            labels=False, radius=0.45, shared_range=False):
    positions = np.asarray(positions) if shared_range else normalized_positions(positions)
    coords = positions * min(width, height) + np.array(center)
    for u, v in edges:
        ax.plot(coords[[u, v], 0], coords[[u, v], 1], color=P['edge'],
                lw=0.35 if not labels else 0.60, zorder=6, solid_capstyle='round',
                gid=f'{name}_edge_{u}_{v}')
    for node, (x, y) in enumerate(coords):
        patch = Circle((x, y), radius, facecolor=COMMUNITY_COLORS[communities[node]],
                       edgecolor=P['white'], linewidth=0.25, zorder=7)
        patch.set_gid(f'{name}_node_{node}')
        ax.add_patch(patch)
        if labels:
            numeral = str(node + 1)
            ink = TextPath((0, 0), numeral,
                           prop=font_manager.FontProperties(family=FONT, size=7.0)).get_extents()
            label = text(ax, x, y, numeral, size=7.0, gid=f'{name}_node_id_{node}')
            label.set_ha('left')
            label.set_va('baseline')
            label.set_position((x - (ink.x0 + ink.x1) * 0.5 * 25.4 / 72,
                                y - (ink.y0 + ink.y1) * 0.5 * 25.4 / 72))
    GLYPHS.append({'name': name, 'node_ids': list(range(len(positions))),
                   'edges': sorted([sorted(e) for e in edges]),
                   'communities': list(communities), 'coordinates_mm': coords.tolist(),
                   'node_radius_mm': radius, 'shared_coordinate_range': shared_range})


def top_panel(ax, graph, communities, layouts):
    box(ax, 'framework', 41, 66, 99, 48, fill='blue', dashed=True,
        edge=P['border'], radius=1.6)
    text(ax, 43.2, 112.0, 'a', size=8.2, weight='bold', ha='left')
    text(ax, 47, 112.0, 'HCTF-GD Framework', size=8.0, weight='bold', ha='left')
    box(ax, 'backbone_lane', 43, 89.0, 95.5, 21.0, fill='blue', dashed=True,
        edge=P['border'], radius=1.2, lw=0.55)
    box(ax, 'proposal_lane', 43, 66.5, 95.5, 21.0, fill='blue', dashed=True,
        edge=P['border'], radius=1.2, lw=0.55)
    text(ax, 45, 108.2, 'Backbone Pathway', size=7.4, weight='bold', ha='left')
    text(ax, 45, 85.7, 'HCTF-enhanced Pathway', size=7.4, weight='bold', ha='left')
    box(ax, 'input_graph_frame', 4.2, 83.9, 15.6, 15.6, fill='white')
    network(ax, 'input_graph', layouts['input'], graph.edges, communities,
            (12, 91.7), 13, 13, radius=0.23, shared_range=True)
    text(ax, 12, 80.0, 'Input Graph', size=7.0)
    text(ax, 12, 75.8, r'$G$', size=10.0)
    box(ax, 'topology_features', 22, 78.2, 15.3, 27,
        'Topology–\nCommunity\nFeatures', size=7.0)
    route(ax, 'graph_to_features', [(19.8, 91.7), (22, 91.7)])
    route(ax, 'features_split', [(37.3, 91.7), (39, 91.7)], arrow=False)
    junction(ax, 39, 91.7)
    for name, cy, title in [('backbone', 101.8, 'Backbone'), ('proposal', 79.2, 'Proposal')]:
        route(ax, name + '_input', [(39, 91.7), (39, cy), (45, cy)])
        box(ax, name + '_block_first', 45, cy - 3.9, 13.5, 7.8,
            'DeepGD\nBlock', fill='cyan', size=7.0)
        box(ax, name + '_block_last', 81, cy - 3.9, 13.5, 7.8,
            'DeepGD\nBlock', fill='cyan', size=7.0)
        box(ax, name + '_coordinate_head', 99, cy - 3.9, 18.6, 7.8,
            'Coordinate\nHead', fill='orange', size=7.0)
        if name == 'backbone':
            route(ax, 'backbone_to_ellipsis', [(58.5, cy), (63, cy)])
            text(ax, 70.2, cy + 0.25, '···', size=9)
            route(ax, 'ellipsis_to_last_block', [(76.3, cy), (81, cy)])
        else:
            box(ax, 'hctf_insertion', 63.5, cy - 3.9, 13.5, 7.8,
                'HCTF', fill='purple', size=7.2, weight='bold')
            route(ax, 'proposal_to_hctf', [(58.5, cy), (63.5, cy)])
            route(ax, 'hctf_to_proposal_block', [(77, cy), (81, cy)])
        route(ax, name + '_to_head', [(94.5, cy), (99, cy)])
        box(ax, name + '_layout_frame', 120.2, cy - 7.8, 15.6, 15.6, fill='white')
        network(ax, name + '_layout', layouts[name], graph.edges, communities,
                (128, cy), 13, 13, radius=0.23, shared_range=True)
        route(ax, name + '_head_to_layout', [(117.6, cy), (120.2, cy)])
        text(ax, 128, cy - 10.3, r'$X_0$' if name == 'backbone' else r'$X_1$',
             size=10.0, gid=name + '_caption_variable')
    box(ax, 'safeblend', 142.65, 87.7, 15.2, 8.0, 'SafeBlend', size=7.0, weight='bold')
    route(ax, 'x0_to_safeblend', [(135.8, 101.8), (141.5, 101.8), (141.5, 93.5), (142.65, 93.5)])
    route(ax, 'x1_to_safeblend', [(135.8, 79.2), (141.5, 79.2), (141.5, 89.8), (142.65, 89.8)])
    box(ax, 'final_graph_frame', 160.2, 83.9, 15.6, 15.6, fill='white')
    network(ax, 'final_layout', layouts['final'], graph.edges, communities,
            (168, 91.7), 13, 13, radius=0.23, shared_range=True)
    route(ax, 'safeblend_to_final', [(157.85, 91.7), (160.2, 91.7)])
    text(ax, 168, 80.0, 'Final Layout', size=7.0)
    text(ax, 168, 75.8, r'$X^*$', size=10.0)
    route(ax, 'enlargement_left', [(66.0, 76.1), (66.0, 64.7), (5, 61)],
          arrow=False, dashed=True, lw=0.5, color=P['border'], z=2)
    route(ax, 'enlargement_right', [(74.5, 76.1), (74.5, 64.7), (175, 61)],
          arrow=False, dashed=True, lw=0.5, color=P['border'], z=2)


def bottom_panel(ax, graph, communities, audit):
    box(ax, 'hctf_block_enlarged', 3.3, 2, 173.4, 59, fill='blue', dashed=True,
        edge=P['border'], radius=1.6)
    text(ax, 5.3, 58.5, 'b', size=8.2, weight='bold', ha='left')
    text(ax, 9.4, 58.5, 'HCTF Block', size=8.0, weight='bold', ha='left')
    text(ax, 5.8, 35, r'$h_i$', size=10.0)
    route(ax, 'raw_h_to_norm', [(8.1, 35), (11, 35)])
    box(ax, 'layer_norm', 11, 31.1, 15, 7.8, 'LayerNorm', size=7.0)
    route(ax, 'norm_to_junction', [(26, 35), (27.4, 35)], arrow=False)
    junction(ax, 27.4, 35)
    route(ax, 'normalized_to_node', [(27.4, 35), (27.4, 51.5), (32, 51.5)])
    for key, x, w, value in [('node_linear1', 32, 12, 'Linear'),
                           ('node_silu', 47, 10, 'SiLU'),
                           ('node_dropout', 60, 15, 'Dropout'),
                           ('node_linear2', 78, 12, 'Linear')]:
        box(ax, key, x, 47.6, w, 7.8, value, size=7.0)
    for key, x0, x1 in [('linear1_to_silu', 44, 47), ('silu_to_dropout', 57, 60),
                        ('dropout_to_linear2', 75, 78), ('linear2_to_u', 90, 94)]:
        route(ax, key, [(x0, 51.5), (x1, 51.5)])
    box(ax, 'node_residual', 94, 46.7, 19, 9.6, fill='purple')
    text(ax, 103.5, 53.5, 'Node Residual', size=7.0, owner='node_residual')
    text(ax, 103.5, 49.7, r'$u_i$', size=10.0, owner='node_residual')
    route(ax, 'normalized_hidden_bus', [(27.4, 35), (27.4, 45.2), (104, 45.2)], arrow=False)
    text(ax, 24.2, 44.8, r'$\bar{h}_i$', size=10.0)
    junction(ax, 89.5, 45.2)
    route(ax, 'hidden_to_bidirectional', [(89.5, 45.2), (89.5, 35.7)])
    route(ax, 'hidden_to_structural', [(104, 45.2), (104, 40.5), (117.5, 40.5), (117.5, 37)])
    box(ax, 'complementary_forests', 31, 25, 45, 19, fill='white', edge='#9BA7B0')
    text(ax, 53.5, 41.6, 'Complementary Forests', size=7.4, weight='bold', owner='complementary_forests')
    base = np.array([[-5.4, 0], [-3.3, 4.05], [-3.3, -4.05], [0, 4.05],
                     [0, -4.05], [3.3, 4.05], [2.4, 0], [3.3, -4.05], [5.4, 0]], dtype=float)
    for t, cx in enumerate((38.5, 53.5, 68.5)):
        network(ax, f'forest_{t + 1}', base, audit['forests'][t], communities,
                (cx, 34.2), 10.8, 10.8, labels=True, radius=1.10)
        text(ax, cx, 27.1, f'F{chr(0x2081 + t)}', size=7.2)
    # A single graph-input bus. Only filled dots indicate electrical-style
    # junctions; continuous crossings without dots are independent signals.
    # The topology-only forest constructor also uses b to separate graphs.
    text(ax, 17.0, 19.5, 'Graph inputs\nG, c, b', size=7.0, gid='graph_input_label')
    route(ax, 'graph_input_source', [(25.35, 19.5), (25.8, 19.5)], arrow=False)
    route(ax, 'graph_input_bus', [(25.8, 11), (25.8, 30)], arrow=False)
    for jy in (11, 19.5, 23, 30):
        junction(ax, 25.8, jy)
    route(ax, 'graph_to_forest', [(25.8, 30), (29.1, 30), (29.1, 31), (31, 31)])
    text(ax, 21.2, 28.4, 'G, b', size=7.0, gid='forest_input_label')
    box(ax, 'bidirectional_aggregation', 80, 26.3, 19, 9.4, 'Bidirectional\nAggregation', fill='cyan', size=7.0)
    route(ax, 'forest_to_aggregation', [(76, 31), (80, 31)])
    route(ax, 'graph_to_context', [(25.8, 23), (78.2, 23), (80, 23)])
    junction(ax, 78.2, 23)
    route(ax, 'relations_to_aggregation', [(78.2, 23), (78.2, 38.9), (84.5, 38.9), (84.5, 35.7)], lw=0.55)
    text(ax, 83, 41.8, 'G, c', size=7.0)
    box(ax, 'community_context', 80, 17.8, 19, 7.6, 'Community\nContext', fill='cyan', size=7.0)
    route(ax, 'hidden_to_community', [(27.4, 35), (27.4, 21), (80, 21)])
    junction(ax, 27.4, 21)
    box(ax, 'structural_residual', 108, 25, 19, 12, fill='purple')
    text(ax, 117.5, 33.4, 'Structural\nResidual', size=7.0, owner='structural_residual')
    text(ax, 117.5, 27.8, r'$v_i$', size=10.0, owner='structural_residual')
    route(ax, 'aggregate_to_structural', [(99, 31), (108, 31)])
    route(ax, 'context_to_structural', [(99, 21.6), (104, 21.6), (104, 27.5), (108, 27.5)])
    route(ax, 'graph_to_descriptor', [(25.8, 11), (31, 11)])
    route(ax, 'hidden_to_descriptor', [(27.4, 21), (27.4, 17.5), (43, 17.5), (43, 16.25)])
    box(ax, 'graph_descriptor', 31, 5.75, 24, 10.5, fill='cyan')
    text(ax, 43, 13.2, 'Graph Descriptor', size=7.0, owner='graph_descriptor')
    text(ax, 43, 8.8, r'$d_G$', size=10.0, owner='graph_descriptor')
    box(ax, 'gate_mlp', 60, 7.0, 18, 8.0, r'$\mathrm{MLP}_{\mathrm{gate}}$', size=10.0)
    box(ax, 'gate_sigmoid', 82, 7.0, 8, 8.0, r'$\sigma$', size=10.0)
    box(ax, 'adaptive_gate', 94, 5.75, 21, 10.5, fill='orange')
    text(ax, 104.5, 13.2, 'Adaptive Gate', size=7.0, owner='adaptive_gate')
    text(ax, 104.5, 8.8, r'$g_G$', size=10.0, owner='adaptive_gate')
    for name, x0, x1 in [('descriptor_to_mlp', 55, 60), ('mlp_to_sigma', 78, 82),
                        ('sigma_to_graph_gate', 90, 94)]:
        route(ax, name, [(x0, 11), (x1, 11)])
    # Same arithmetic; preserve internal output clearance after insetting the frame.
    op(ax, 'multiply_g_v', 132, 31, '×')
    op(ax, 'add_u', 141.2, 31, '+')
    op(ax, 'multiply_scale', 151.0, 31, '×')
    op(ax, 'add_identity', 161.5, 31, '+')
    route(ax, 'v_to_product', [(127, 31), (129.95, 31)])
    route(ax, 'g_to_product', [(115, 11), (116, 11), (116, 19), (132, 19), (132, 28.95)])
    route(ax, 'gv_to_add_u', [(134.05, 31), (139.15, 31)])
    route(ax, 'u_to_add_u', [(113, 51.5), (141.2, 51.5), (141.2, 33.05)])
    route(ax, 'sum_to_scale', [(143.25, 31), (148.95, 31)])
    box(ax, 'node_modulation_and_scale', 144.0, 40.2, 14, 8.6, r'$s\,a_i$', size=10.0)
    route(ax, 'scale_to_product', [(151.0, 40.2), (151.0, 33.05)])
    route(ax, 'scaled_to_identity', [(153.05, 31), (159.45, 31)])
    junction(ax, 8.5, 35)
    route(ax, 'raw_identity_skip', [(8.5, 35), (8.5, 4.5), (161.5, 4.5), (161.5, 28.95)])
    box(ax, 'updated_hidden', 166.0, 26.7, 7.0, 8.6, r"$h_i'$", fill='purple', size=10.0)
    route(ax, 'identity_to_output', [(163.55, 31), (166.0, 31)])


def audit_geometry(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    outside = []
    owner_overflow = []
    for obj, owner in TEXTS:
        bbox = obj.get_window_extent(renderer).transformed(ax.transData.inverted())
        row = {'id': obj.get_gid(), 'text': obj.get_text(), 'bbox_mm': list(bbox.extents), 'owner': owner}
        bounds.append(row)
        if bbox.x0 < -0.05 or bbox.x1 > WIDTH_MM + 0.05 or bbox.y0 < -0.05 or bbox.y1 > HEIGHT_MM + 0.05:
            outside.append(row)
        if owner:
            x, y, w, h = BOXES[owner]
            if bbox.x0 < x + 0.2 or bbox.x1 > x + w - 0.2 or bbox.y0 < y + 0.1 or bbox.y1 > y + h - 0.1:
                owner_overflow.append(row)
    collisions = []
    for i, left in enumerate(bounds):
        for right in bounds[i + 1:]:
            a, b = left['bbox_mm'], right['bbox_mm']
            if min(a[2], b[2]) - max(a[0], b[0]) > 0.1 and min(a[3], b[3]) - max(a[1], b[1]) > 0.1:
                collisions.append([left['id'], right['id']])
    # Test all ordinary connectors against all text rectangles. Their intended
    # endpoints terminate outside box text, so no hidden pass-through is allowed.
    line_text = []
    for connector in ROUTES:
        if connector['dashed']:
            continue
        path = MplPath(connector['points_mm'])
        for label in bounds:
            x0, y0, x1, y1 = label['bbox_mm']
            if path.intersects_bbox(Bbox.from_extents(x0 - 0.08, y0 - 0.08, x1 + 0.08, y1 + 0.08), filled=False):
                line_text.append([connector['name'], label['id']])
    return {'text_outside_canvas': outside, 'text_outside_owner': owner_overflow,
            'text_text_collisions': collisions, 'connector_text_intersections': line_text,
            'text_bounds': bounds, 'connectors': ROUTES}


def embed_svg_fonts(filename):
    """Embed small TrueType subsets without outlining any editable SVG text."""
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(filename)
    root = tree.getroot()
    # Matplotlib serializes the generic family as quoted 'sans-serif', which
    # browsers can interpret as a literal missing font and render with a serif
    # fallback. Bind editable SVG text to the exact embedded PDF font family.
    for node in root.iter():
        if node.tag in (f'{{{ns}}}text', f'{{{ns}}}tspan') and 'style' in node.attrib:
            node.set('style', node.get('style').replace("'sans-serif'", "'DejaVu Sans'"))
    characters = ''.join(''.join(t.itertext()) for t in root.findall(f'.//{{{ns}}}text'))
    rules = []
    for weight, style in [('normal', 'normal'), ('bold', 'normal'), ('normal', 'oblique')]:
        prop = font_manager.FontProperties(family=FONT, weight=weight, style=style)
        options = subset.Options()
        options.drop_tables += ['FFTM']
        font = subset.load_font(font_manager.findfont(prop), options)
        worker = subset.Subsetter(options=options)
        worker.populate(text=characters)
        worker.subset(font)
        buffer = BytesIO()
        font.save(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        rules.append(f"@font-face {{font-family:'DejaVu Sans';font-style:{style};font-weight:{weight};src:url(data:font/ttf;base64,{encoded}) format('truetype');}}")
    defs = root.find(f'{{{ns}}}defs')
    style_node = ET.SubElement(defs, f'{{{ns}}}style', {'type': 'text/css', 'id': 'embedded_editable_fonts'})
    style_node.text = '\n'.join(rules)
    tree.write(filename, encoding='utf-8', xml_declaration=True)


def main():
    if not (OUT / 'MODEL_AUDIT.json').exists():
        raise FileNotFoundError('MODEL_AUDIT.json must accompany this drawing source')
    audit = json.loads((OUT / 'MODEL_AUDIT.json').read_text(encoding='utf-8'))
    graph = nx.Graph()
    graph.add_nodes_from(audit['nodes'])
    graph.add_edges_from(audit['edges'])
    communities = audit['communities']
    input_edges = {tuple(sorted(e)) for e in graph.edges}
    forest_checks = []
    for edges in audit['forests']:
        assert len(edges) == len({tuple(sorted(e)) for e in edges})
        assert all(u != v and u in graph and v in graph for u, v in edges)
        f = nx.Graph()
        f.add_nodes_from(graph.nodes)
        f.add_edges_from(edges)
        assert nx.is_forest(f)
        assert {tuple(sorted(e)) for e in edges} <= input_edges
        assert list(sorted(map(sorted, nx.connected_components(f)))) == list(sorted(map(sorted, nx.connected_components(graph))))
        forest_checks.append({'acyclic': True, 'original_edge_subset': True,
                              'nodes': f.number_of_nodes(), 'edges': f.number_of_edges()})
    assert len({tuple(map(tuple, edges)) for edges in audit['forests']}) == 3
    # Read saved coordinates verbatim; never run a layout optimizer or move nodes.
    stage_audit = json.loads((OUT / 'STAGE_VIEW_AUDIT.json').read_text(encoding='utf-8'))
    with np.load(OUT / 'REAL_STAGE_LAYOUTS.npz', allow_pickle=False) as saved:
        top_nodes = saved['nodes'].tolist()
        top_edges = saved['edges'].tolist()
        top_communities = saved['communities'].tolist()
        layouts = {'backbone': saved['x0'].copy(), 'proposal': saved['x1'].copy(),
                   'final': saved['final'].copy()}
    top_graph = nx.Graph()
    top_graph.add_nodes_from(top_nodes)
    top_graph.add_edges_from(top_edges)
    assert top_nodes == list(range(48)) and top_graph.number_of_edges() == 67
    assert len(top_communities) == 48 and len(set(top_communities)) == 6
    view_span = max(float(np.ptp(v, axis=0).max()) for v in layouts.values())
    angles = np.arange(len(top_nodes)) * (2 * np.pi / len(top_nodes)) + np.pi / 2
    layouts['input'] = 0.47 * view_span * np.column_stack((np.cos(angles), np.sin(angles)))
    view_centers = {k: (v.max(axis=0) + v.min(axis=0)) / 2 for k, v in layouts.items()}
    display_layouts = {k: (v - view_centers[k]) / view_span for k, v in layouts.items()}
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH_MM), ylim=(0, HEIGHT_MM), aspect='equal')
    ax.set_axis_off()
    top_panel(ax, top_graph, top_communities, display_layouts)
    bottom_panel(ax, graph, communities, audit)
    geometry = audit_geometry(fig, ax)
    for check in ('text_outside_canvas', 'text_outside_owner',
                  'text_text_collisions', 'connector_text_intersections'):
        assert not geometry[check], (check, geometry[check])
    # Explicit canvas margins preserve the physical contract in every format.
    fig.savefig(OUT / f'{STEM}.svg')
    embed_svg_fonts(OUT / f'{STEM}.svg')
    fig.savefig(OUT / f'{STEM}.pdf')
    fig.savefig(OUT / f'{STEM}.png', dpi=600)
    # RGB export drops only the fully opaque alpha channel; no pixel appearance changes.
    with Image.open(OUT / f'{STEM}.png') as rgba:
        rgb = rgba.convert('RGB')
        rgb.save(OUT / f'{STEM}.tiff', dpi=(600, 600), compression='raw')
    fig.savefig(OUT / 'QA_full_180mm.png', dpi=300)
    fig.savefig(OUT / 'QA_top_detail.png', dpi=300,
                bbox_inches=Bbox.from_extents(0, 65 / 25.4, 180 / 25.4, 116 / 25.4))
    fig.savefig(OUT / 'QA_block_detail.png', dpi=300,
                bbox_inches=Bbox.from_extents(0, 1 / 25.4, 180 / 25.4, 64 / 25.4))
    fig.savefig(OUT / 'QA_fusion_detail.png', dpi=600,
                bbox_inches=Bbox.from_extents(77.9 / 25.4, 3 / 25.4, 178 / 25.4, 57 / 25.4))
    plt.close(fig)
    four = [g for g in GLYPHS if g['name'] in ('input_graph', 'backbone_layout', 'proposal_layout', 'final_layout')]
    assert len(four) == 4
    assert all(g['node_ids'] == four[0]['node_ids'] and g['edges'] == four[0]['edges']
               and g['communities'] == four[0]['communities'] for g in four)
    assert all(g['node_radius_mm'] == 0.23 and g['shared_coordinate_range'] for g in four)
    frame_names = ['input_graph_frame', 'backbone_layout_frame', 'proposal_layout_frame', 'final_graph_frame']
    assert all(BOXES[k][2:] == (15.6, 15.6) for k in frame_names)
    for name in ('backbone', 'proposal'):
        x, y, w, h = BOXES[name + '_layout_frame']
        lx, ly, lw, lh = BOXES[name + '_lane']
        assert x > lx and x + w < lx + lw and y > ly and y + h < ly + lh
    nonincident_hits = []
    for glyph in GLYPHS:
        if not glyph['name'].startswith('forest_'):
            continue
        xy = np.asarray(glyph['coordinates_mm'])
        for u, v in glyph['edges']:
            edge_vector = xy[v] - xy[u]
            denom = float(edge_vector @ edge_vector)
            for node in set(glyph['node_ids']) - {u, v}:
                fraction = np.clip(float((xy[node] - xy[u]) @ edge_vector) / denom, 0, 1)
                distance = float(np.linalg.norm(xy[node] - (xy[u] + fraction * edge_vector)))
                if distance < glyph['node_radius_mm'] + 0.18:
                    nonincident_hits.append([glyph['name'], u, v, node])
    assert nonincident_hits == [], nonincident_hits
    output_right = BOXES['updated_hidden'][0] + BOXES['updated_hidden'][2]
    container_right = BOXES['hctf_block_enlarged'][0] + BOXES['hctf_block_enlarged'][2]
    output_clearance = container_right - output_right
    assert output_clearance >= 3.0
    caption_clearances = {}
    for name in ('backbone', 'proposal'):
        label_variable = next(b for b in geometry['text_bounds'] if b['id'] == name + '_caption_variable')
        image_gap = BOXES[name + '_layout_frame'][1] - label_variable['bbox_mm'][3]
        lane_gap = label_variable['bbox_mm'][1] - BOXES[name + '_lane'][1]
        assert min(image_gap, lane_gap) >= 0.6
        caption_clearances[name] = {'image_to_caption_mm': image_gap, 'caption_to_lane_mm': lane_gap}
    assert min(obj.get_fontsize() for obj, _ in TEXTS) >= 7.0
    body_boxes = {k: v for k, v in BOXES.items()
                  if k not in ('framework', 'backbone_lane', 'proposal_lane', 'hctf_block_enlarged')}
    box_collisions = []
    box_items = list(body_boxes.items())
    for index, (left_name, (lx, ly, lw, lh)) in enumerate(box_items):
        for right_name, (rx, ry, rw, rh) in box_items[index + 1:]:
            if min(lx + lw, rx + rw) - max(lx, rx) > 0.05 and min(ly + lh, ry + rh) - max(ly, ry) > 0.05:
                box_collisions.append([left_name, right_name])
    assert box_collisions == [], box_collisions
    svg = ET.parse(OUT / f'{STEM}.svg').getroot()
    ns = {'s': 'http://www.w3.org/2000/svg'}
    geometry.update({
        'physical_size_mm': [WIDTH_MM, HEIGHT_MM], 'forest_checks': forest_checks,
        'four_thumbnails_same_identity_topology_communities': True,
        'svg_text_elements': len(svg.findall('.//s:text', ns)),
        'svg_embedded_image_elements': len(svg.findall('.//s:image', ns)),
        'schematic_layouts': False,
        'input_topology_view': 'Circular presentation; not an experimental input coordinate array',
        'final_thumbnail': 'Archived accepted SafeBlend layout, alpha=1.0',
        'stage_provenance': stage_audit,
        'thumbnail_frame_sizes_mm': [list(BOXES[k][2:]) for k in frame_names],
        'thumbnail_node_radius_mm': 0.23,
        'common_coordinate_view': {'centers': {k: v.tolist() for k, v in view_centers.items()},
                                   'span': view_span, 'drawing_rotation': False},
        'forest_nonincident_edge_node_collisions': nonincident_hits,
        'output_to_right_border_clearance_mm': output_clearance,
        'caption_clearances_mm': caption_clearances,
        'minimum_declared_text_size_pt': min(obj.get_fontsize() for obj, _ in TEXTS),
        'module_border_collisions': box_collisions,
        'community_to_gate_border_gap_mm': BOXES['community_context'][1] - (BOXES['adaptive_gate'][1] + BOXES['adaptive_gate'][3]),
        'layout_data': {k: v.tolist() for k, v in layouts.items()},
        'network_glyphs': GLYPHS,
        'png': {}, 'tiff': {},
    })
    for extension, key in [('png', 'png'), ('tiff', 'tiff')]:
        with Image.open(OUT / f'{STEM}.{extension}') as img:
            geometry[key] = {'pixels': list(img.size), 'dpi': [float(d) for d in img.info.get('dpi', ())], 'mode': img.mode}
    (OUT / 'GEOMETRY_QA.json').write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    (OUT / 'FIGURE_SOURCE_DATA.json').write_text(json.dumps({
        'data_kind': 'Archived real output layouts; circular input topology view',
        'nodes': top_nodes, 'edges': top_edges, 'communities': top_communities,
        'coordinates': geometry['layout_data'],
        'common_coordinate_view': geometry['common_coordinate_view'],
        'stage_provenance': stage_audit,
        'lower_forest_example': {'nodes': audit['nodes'], 'edges': audit['edges'],
                                'communities': communities, 'forests': audit['forests'],
                                'construction': audit['forest_constructor']},
    }, indent=2), encoding='utf-8')
    print(json.dumps({k: geometry[k] for k in ('text_outside_canvas', 'text_outside_owner',
           'text_text_collisions', 'connector_text_intersections', 'svg_text_elements',
           'svg_embedded_image_elements', 'png', 'tiff')}, indent=2))


if __name__ == '__main__':
    main()
