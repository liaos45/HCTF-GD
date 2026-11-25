"""Standalone reproduction source for the audited, unchanged forest graphic.

Formal delivery uses verify_statistics.py to copy the verified existing
graphic byte-for-byte. This source was tested in a temporary directory with
pixel-identical PNG output. Run it when a fresh render is needed; it uses
the same archived tables, axes, typography and original plotting geometry.
"""
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



from matplotlib.lines import Line2D
import numpy as np

BLUE, GREEN, ORANGE, GRAY = "#638FB5", "#689C83", "#C89561", "#75818F"
DOMAINS = ["enzymes", "fb15k237", "imdb_binary", "north", "proteins", "rome", "sbm", "wn18rr"]
NAMES = {"enzymes": "ENZYMES", "fb15k237": "FB15K237", "imdb_binary": "IMDB-BINARY", "north": "NORTH",
         "proteins": "PROTEINS", "rome": "ROME", "sbm": "SBM", "wn18rr": "WN18RR", "ALL": "Equal-domain mean"}
mpl.rcParams.update({"axes.linewidth": .65, "axes.titlesize": 9.2})


def interval(ax, mean, low, high, y, color, marker):
    ax.hlines(y, low, high, color=color, lw=1., zorder=2)
    ax.plot([low, low], [y-.055, y+.055], color=color, lw=.8)
    ax.plot([high, high], [y-.055, y+.055], color=color, lw=.8)
    ax.plot(mean, y, marker, color=color, ms=3.8, zorder=3)


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#E5E9ED", lw=.5, zorder=0)
    ax.tick_params(length=2.5, width=.6)


def main():
    fig = figure(106)
    text(fig, 4, 100, "(a) Shared warm-start: eight domains", 9.1, weight="bold")
    text(fig, 107, 100, "(b) Strict size-OOD", 9.1, weight="bold")
    ax = axis(fig, 28, 24, 66, 67)
    frame = read("main_confidence_archive.csv")
    frame = frame[frame.metric == "graph_stress"]
    order = DOMAINS+["ALL"]
    for i, domain in enumerate(order):
        y = len(order)-1-i
        for comparison, color, marker, offset in (("full_vs_node", BLUE, "o", .14),
                                                   ("full_vs_backbone", GREEN, "s", -.14)):
            row = frame[(frame.dataset == domain) & (frame.comparison == comparison)].iloc[0]
            interval(ax, row.mean_gain_pct, row.ci_low_pct, row.ci_high_pct, y+offset,
                     ORANGE if domain == "ALL" else color, marker)
    ax.axvline(0, color=GRAY, lw=.7)
    ax.set_yticks(np.arange(9), [NAMES[d] for d in order][::-1])
    ax.set_xlim(-15, 25); ax.set_ylim(-.6, 8.6)
    ax.set_xlabel("Stress improvement (%)"); clean(ax)
    ax = axis(fig, 123, 24, 52, 67)
    frame = read("strict_confidence_archive.csv")
    frame = frame[(frame.metric == "graph_stress") & (frame.comparison == "full_raw_vs_node_raw")]
    order = ["enzymes", "fb15k237", "sbm", "ALL"]
    for i, domain in enumerate(order):
        row = frame[frame.dataset == domain].iloc[0]
        interval(ax, row.mean_gain_pct, row.ci_low_pct, row.ci_high_pct, 3-i,
                 ORANGE if domain == "ALL" else BLUE, "D" if domain == "ALL" else "o")
    ax.axvline(0, color=GRAY, lw=.7)
    ax.set_yticks(np.arange(4), [NAMES[d] for d in order][::-1]); ax.set_ylim(-.5, 3.5)
    ax.set_xlim(-.16, 1.23); ax.set_xlabel("Stress improvement (%)"); clean(ax)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1, label="Full vs Node-only"),
                        Line2D([], [], color=GREEN, marker="s", lw=1, label="Full vs Matching Backbone")],
               loc="lower left", bbox_to_anchor=(.05, .005), ncol=2, frameon=False)
    save(fig, "Fig05_Paired_Stress_Effects")


if __name__ == "__main__":
    main()
