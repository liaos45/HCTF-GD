from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


class FlingUnavailableError(RuntimeError):
    pass


class FlingAdapter:
    def __init__(self, root="fling-layout-main", device=None):
        self.root = Path(root).resolve()
        self.device = device

    def _classes(self):
        if not self.root.exists():
            raise FlingUnavailableError(f"Fling source directory not found: {self.root}")
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from fling import FlingBlend, FlingStress, FlingVis
        except Exception as error:
            raise FlingUnavailableError(f"could not import Fling from {self.root}: {error}") from error
        return {
            "flingstress": FlingStress,
            "flingvis": FlingVis,
            "flingblend": FlingBlend,
        }

    @staticmethod
    def graph_from_edge_index(num_nodes, edge_index):
        graph = nx.Graph()
        graph.add_nodes_from(range(int(num_nodes)))
        graph.add_edges_from((int(u), int(v)) for u, v in edge_index.detach().cpu().t().tolist())
        return graph

    def layout(self, method, graph, seed=0, iterations=None, blend_lambda=0.5):
        method = str(method).lower().replace("_", "")
        classes = self._classes()
        if method not in classes:
            raise ValueError(f"unknown Fling method: {method}")
        kwargs = {"n_components": 2, "seed": int(seed)}
        if iterations is not None:
            kwargs["iters"] = int(iterations)
        if method in {"flingvis", "flingblend"} and self.device is not None:
            kwargs["device"] = self.device
        estimator = classes[method](**kwargs)
        with torch.enable_grad():
            if method == "flingblend":
                pos = estimator.fit_transform(graph, lam=float(blend_lambda))
            else:
                pos = estimator.fit_transform(graph)
        return np.asarray(pos, dtype=np.float32)
