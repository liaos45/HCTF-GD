from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
from typing import Iterator, Optional, TypeVar

import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from tqdm.auto import tqdm


DATATYPE = TypeVar("DATATYPE", bound=BaseData)


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class SBMDataset(pyg.data.InMemoryDataset):
    DEFAULT_NAME = "SBM"

    def __init__(
        self,
        *,
        root: str = DATASET_ROOT,
        name: str = DEFAULT_NAME,
        index: Optional[list[str]] = None,
        num_graphs: int = 1500,
        min_nodes: int = 30,
        max_nodes: int = 90,
        min_communities: int = 3,
        max_communities: int = 8,
        seed: int = 42,
        datatype: type[DATATYPE] = GraphDrawingData,
    ):
        self.dataset_name = name
        self.index = index
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_communities = min_communities
        self.max_communities = max_communities
        self.seed = seed
        self.datatype = datatype
        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter,
        )
        self.data, self.slices = torch_load_trusted(self.data_path)
        with open(self.index_path, "r") as index_file:
            self.index = index_file.read().strip().split("\n")
        data_list = map(datatype.static_transform, tqdm(self, desc="Transform graphs"))
        data_dict = {data.G.graph["name"]: data for data in data_list}
        data_list = [data_dict[name] for name in self.index]
        self.data, self.slices = self.collate(list(data_list))

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]

    def _community_sizes(self, rng: np.random.Generator, n: int, c: int) -> list[int]:
        weights = rng.dirichlet(np.ones(c) * 1.4)
        sizes = np.maximum(3, np.round(weights * n).astype(int))
        while sizes.sum() != n:
            diff = n - int(sizes.sum())
            idx = int(rng.integers(0, c))
            if diff > 0:
                sizes[idx] += 1
            elif sizes[idx] > 3:
                sizes[idx] -= 1
        return sizes.tolist()

    def _make_graph(self, graph_id: int, rng: np.random.Generator) -> nx.Graph:
        n = int(rng.integers(self.min_nodes, self.max_nodes + 1))
        c = int(rng.integers(self.min_communities, min(self.max_communities, n // 3) + 1))
        sizes = self._community_sizes(rng, n, c)

        p_in = float(rng.uniform(0.16, 0.34))
        p_out = float(rng.uniform(0.006, 0.035))
        probs = np.full((c, c), p_out, dtype=float)
        np.fill_diagonal(probs, p_in)

        G = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(0, 2**31 - 1)))
        G.remove_edges_from(nx.selfloop_edges(G))

        offset = 0
        community_nodes = []
        for comm_id, size in enumerate(sizes):
            nodes = list(range(offset, offset + size))
            community_nodes.append(nodes)
            for node in nodes:
                G.nodes[node]["community"] = comm_id
            offset += size

        # Keep every graph connected without destroying the block structure.
        components = [list(comp) for comp in nx.connected_components(G)]
        while len(components) > 1:
            left = components.pop()
            right = components[0]
            G.add_edge(int(rng.choice(left)), int(rng.choice(right)))
            components = [list(comp) for comp in nx.connected_components(G)]

        # Add a sparse backbone between adjacent communities so CAID/CPSG see
        # realistic boundary nodes without dense cross-community edges.
        for comm_id in range(c - 1):
            if rng.random() < 0.8:
                u = int(rng.choice(community_nodes[comm_id]))
                v = int(rng.choice(community_nodes[comm_id + 1]))
                G.add_edge(u, v)

        G.graph.update(
            dict(
                name=f"sbm_{graph_id:05d}",
                dataset=self.dataset_name,
            )
        )
        return G

    def generate(self) -> Iterator[nx.Graph]:
        rng = np.random.default_rng(self.seed)
        for graph_id in tqdm(range(self.num_graphs), desc="Generate SBM graphs"):
            yield self._make_graph(graph_id, rng)

    def download(self) -> None:
        return None

    def process(self) -> None:
        def filter_and_save_index(data_list):
            name_list = []
            for data in data_list:
                if self.pre_filter(data):
                    name_list.append(data.G.graph["name"])
                    yield data
            if self.index is None:
                self.index = name_list
            else:
                self.index = [name for name in self.index if name in name_list]
            with open(self.index_path, "w") as index_file:
                index_file.write("\n".join(self.index))

        data_list = map(self.datatype, self.generate())
        data_list = filter_and_save_index(data_list)
        data_list = map(self.pre_transform, data_list)
        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
