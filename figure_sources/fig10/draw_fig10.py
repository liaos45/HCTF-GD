"""Render audited real-layout comparisons only, at native 180 mm width.

No inference, coordinate optimization, sampling or dropped edges. Plot data
use proper rigid rotations and a common isotropic display scale per graph;
every reported metric comes from the original evaluation coordinate table.
"""
from pathlib import Path
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
WIDTH_MM = 180.0
INK = "#263343"
EDGE = "#AEB8C2"
ORANGE = "#C89561"
PALETTE = ["#638FB5", "#689C83", "#8977B2", ORANGE, "#8399A2", "#B18D9C", "#9AA570", "#7D92AD"]
METHODS = ["Matching Backbone", "Node-only HCTF", "Raw Full HCTF", "SafeBlend Final"]
CENTERS = [24.5, 68, 111.5, 155]
TRANSFORMS, EGO = [], []
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                    "font.size": 8.2, "axes.labelsize": 8.2, "xtick.labelsize": 8., "ytick.labelsize": 8.,
                    "legend.fontsize": 8., "pdf.fonttype": 42, "svg.fonttype": "none",
                    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def read(name):
    return pd.read_csv(DATA / name)


def figure(height):
    return plt.figure(figsize=(180/25.4, height/25.4), dpi=600)


def text(fig, x, y, value, size=8.2, **kwargs):
    return fig.text(x/180, y/(fig.get_figheight()*25.4), value, color=INK, fontsize=size,
                    ha=kwargs.pop("ha", "left"), va="center", **kwargs)


def axis(fig, x, y, w, h):
    return fig.add_axes([x/180, y/(fig.get_figheight()*25.4), w/180, h/(fig.get_figheight()*25.4)])


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inactive_ticks = set()
    for ax in fig.axes:
        for axis_object, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in axis_object.get_major_ticks()+axis_object.get_minor_ticks():
                if not low-1e-12 <= tick.get_loc() <= high+1e-12:
                    inactive_ticks.update((tick.label1, tick.label2))
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text() or artist in inactive_ticks:
            continue
        bb = artist.get_window_extent(renderer)
        assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.x1+1 and bb.y1 <= fig.bbox.y1+1, artist.get_text()
    for folder in ("pdf", "svg", "png", "preview"):
        (OUT / folder).mkdir(exist_ok=True)
    fig.savefig(OUT / "pdf" / (stem + ".pdf"), facecolor="white")
    fig.savefig(OUT / "svg" / (stem + ".svg"), facecolor="white")
    path = OUT / "png" / (stem + ".png")
    fig.savefig(path, dpi=600, facecolor="white")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.save(path, dpi=(600, 600))
        rgb.save(OUT / "png" / (stem + ".tiff"), dpi=(600, 600), compression="raw")
        w = round(180/25.4*300)
        rgb.resize((w, round(rgb.height*w/rgb.width)), Image.Resampling.LANCZOS).save(
            OUT / "preview" / (stem + "_180mm.png"), dpi=(300, 300))
    plt.close(fig)
    print("[RENDERED]", stem, flush=True)


def case(seed, gid, figure_name):
    nodes = read("layout_nodes.csv")
    nodes = nodes[(nodes.seed == seed) & (nodes.graph_id == gid)]
    edge = read("layout_edges.csv")
    edges = edge[(edge.seed == seed) & (edge.graph_id == gid)][["source", "target"]].to_numpy(int)
    positions = {}
    labels = None
    for method in METHODS:
        group = nodes[nodes.method == method].sort_values("node_id")
        assert group.node_id.tolist() == list(range(len(group)))
        if labels is None:
            labels = group.community.to_numpy(int)
        assert np.array_equal(labels, group.community.to_numpy(int))
        positions[method] = group[["x", "y"]].to_numpy()
    reference = positions[METHODS[0]]-positions[METHODS[0]].mean(0)
    rotated, transforms = {}, []
    for method, xy in positions.items():
        centered = xy-xy.mean(0)
        u, _, vt = np.linalg.svd(centered.T @ reference)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        assert np.linalg.det(rotation) > .999999
        rotated[method] = centered @ rotation
        transforms.append(dict(figure=figure_name, seed=seed, graph_id=gid, method=method,
                               center_x=xy.mean(0)[0], center_y=xy.mean(0)[1], r00=rotation[0, 0],
                               r01=rotation[0, 1], r10=rotation[1, 0], r11=rotation[1, 1]))
    shared_scale = max(np.max(np.abs(xy)) for xy in rotated.values())*1.08
    for row in transforms:
        row["common_scale_divisor"] = float(shared_scale)
    TRANSFORMS.extend(transforms)
    positions = {m: xy/shared_scale for m, xy in rotated.items()}
    metrics = read("layout_metrics.csv")
    metrics = metrics[(metrics.seed == seed) & (metrics.graph_id == gid)].set_index("method")
    return positions, edges, labels, metrics


def graph(ax, xy, edges, labels, boundary=False, subset=None):
    ids = np.arange(len(labels)) if subset is None else np.asarray(subset, int)
    visible = set(ids)
    drawn = np.array([e for e in edges if e[0] in visible and e[1] in visible], int).reshape(-1, 2)
    if len(drawn):
        ax.add_collection(LineCollection(xy[drawn], colors="#A6AFB9", linewidths=.44, alpha=.67, zorder=1))
    ax.scatter(xy[ids, 0], xy[ids, 1], s=13, c=[PALETTE[labels[i] % len(PALETTE)] for i in ids],
               edgecolors="white", linewidths=.28, zorder=3)
    if boundary:
        boundary_ids = {int(i) for a, b in edges if labels[a] != labels[b] for i in (a, b)}
        selected = sorted(visible & boundary_ids)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=23, facecolors="none", edgecolors=ORANGE, linewidths=.55, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def headings(fig, y):
    for x, title in zip(CENTERS, ("Matching\nBackbone", "Node-only HCTF", "Raw Full HCTF", "SafeBlend Final")):
        text(fig, x, y, title, 8.8, ha="center", weight="bold")


def layout_plate(cases, stem, supplementary=False):
    step = 53 if supplementary else 50
    height = 14 + step*len(cases) + 3
    fig = figure(height)
    headings(fig, height-5.2)
    for row, (seed, gid, role) in enumerate(cases):
        y = height-17-row*step
        text(fig, 4, y, "(a)" if row == 0 else "(b)", 8.3, weight="bold")
        positions, edges, labels, metrics = case(seed, gid, stem)
        for x, method in zip(CENTERS, METHODS):
            ax = axis(fig, x-20, y-36, 40, 32)
            graph(ax, positions[method], edges, labels)
            m = metrics.loc[method]
            values = f"S {m.stress_normalized:.4f} | X {m.edge_crossing:.0f}\n"
            if supplementary:
                values += f"C {m.community_compactness:.3f} | Sep/C {m.community_sep_comp_ratio:.3f}\nClearance {m.node_edge_clearance:.3f}"
                ypos = y-43.5
            else:
                values += f"Sep/Comp {m.community_sep_comp_ratio:.3f}"
                ypos = y-42
            text(fig, x, ypos, values, 8., ha="center")
    footnote = "S: pair-normalized Stress; X: crossings"
    save(fig, stem)


def community():
    stem = "community_readability_cases"
    fig = figure(151)
    headings(fig, 146)
    for row, (seed, gid, role) in enumerate(((73, 103, "Accepted; whole-proposal community improvement"),
                                          (17, 13, "Mixed community-metric changes; exact fallback"))):
        y = 133-row*65
        text(fig, 4, y, f"SBM | seed {seed} | graph {gid} | {role}", 8.3, weight="bold")
        positions, edges, labels, metrics = case(seed, gid, stem)
        network = nx.Graph(); network.add_nodes_from(range(len(labels))); network.add_edges_from(edges)
        boundary = [i for i in network if any(labels[i] != labels[j] for j in network[i])]
        root = sorted(boundary, key=lambda i: (-sum(labels[i] != labels[j] for j in network[i]), -network.degree(i), i))[0]
        subset = sorted(int(i) for i in ({root} | set(network[root])))
        zoom_scale = max(np.max(np.abs(positions[m][subset]-positions[m][subset].mean(0))) for m in METHODS)*1.12
        for x, method in zip(CENTERS, METHODS):
            ax = axis(fig, x-20, y-36, 40, 32)
            graph(ax, positions[method], edges, labels, boundary=True)
            m = metrics.loc[method]
            text(fig, x, y-43, f"C {m.community_compactness:.3f} | Sep/C {m.community_sep_comp_ratio:.3f}\nClearance {m.node_edge_clearance:.3f}", 8., ha="center")
            offset = positions[method][subset].mean(0)
            xy = (positions[method]-offset)/zoom_scale
            za = axis(fig, x-12, y-61, 24, 12)
            graph(za, xy, edges, labels, boundary=True, subset=subset)
            for spine in za.spines.values():
                spine.set_visible(True); spine.set_color(EDGE); spine.set_linewidth(.4)
            ego_edges = [(int(a), int(b)) for a, b in edges if a in subset and b in subset]
            EGO.append(dict(seed=seed, graph_id=gid, method=method, root=root, node_ids=json.dumps(subset),
                            edges=json.dumps(ego_edges), boundary_nodes=json.dumps(boundary),
                            common_zoom_divisor=float(zoom_scale), center_x=float(offset[0]), center_y=float(offset[1]),
                            full_view_side_mm=32, inset_side_mm=12))
    text(fig, 90, 2.3, "Orange rings: graph-boundary nodes; clearance: evaluation-only in 10+7", 8., ha="center")
    save(fig, stem)


def main():
    layout_plate([(73, 83, "Accepted; median Stress-gain rank 42/83; alpha 0.75"),
                  (17, 13, "Raw proposal failure; exact fallback")], "Fig07_Qualitative_Layout_Comparison")
    pd.DataFrame(TRANSFORMS).to_csv(OUT / "display_transforms.csv", index=False)


if __name__ == "__main__":
    main()
