from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
from collections import defaultdict, deque
from typing import Iterator, Optional, TypeVar

import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import download_url
from tqdm.auto import tqdm


DATATYPE = TypeVar("DATATYPE", bound=BaseData)


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class _KGSampledDataset(pyg.data.InMemoryDataset):
    DEFAULT_NAME = "KGSampled"
    URL = ""
    TRAIN_FILE = "train.txt"

    def __init__(
        self,
        *,
        root: str = DATASET_ROOT,
        name: Optional[str] = None,
        index: Optional[list[str]] = None,
        num_graphs: int = 2000,
        min_nodes: int = 35,
        max_nodes: int = 90,
        max_edges: int = 360,
        seed: int = 42,
        datatype: type[DATATYPE] = GraphDrawingData,
    ):
        self.dataset_name = name or self.DEFAULT_NAME
        self.index = index
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.seed = seed
        self.datatype = datatype
        super().__init__(
            root=os.path.join(root, self.dataset_name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter,
        )
        self.data, self.slices = torch_load_trusted(self.data_path)
        with open(self.index_path, "r", encoding="utf-8") as index_file:
            self.index = index_file.read().strip().split("\n")
        data_list = map(datatype.static_transform, tqdm(self, desc="Transform graphs"))
        data_dict = {data.G.graph["name"]: data for data in data_list}
        data_list = [data_dict[name] for name in self.index]
        self.data, self.slices = self.collate(list(data_list))

    @property
    def raw_file_names(self) -> list[str]:
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f"{self.URL}/{filename}", self.raw_dir)

    def _read_train_graph(self) -> tuple[dict[int, set[int]], dict[int, str], list[tuple[int, int, str]]]:
        entity_to_id: dict[str, int] = {}
        id_to_entity: dict[int, str] = {}
        adjacency: dict[int, set[int]] = defaultdict(set)
        edge_records: list[tuple[int, int, str]] = []

        def entity_id(entity: str) -> int:
            if entity not in entity_to_id:
                idx = len(entity_to_id)
                entity_to_id[entity] = idx
                id_to_entity[idx] = entity
            return entity_to_id[entity]

        train_path = os.path.join(self.raw_dir, self.TRAIN_FILE)
        with open(train_path, encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                head, relation, tail = parts
                src = entity_id(head)
                dst = entity_id(tail)
                if src == dst:
                    continue
                adjacency[src].add(dst)
                adjacency[dst].add(src)
                edge_records.append((src, dst, relation))

        return adjacency, id_to_entity, edge_records

    def _sample_nodes(
        self,
        adjacency: dict[int, set[int]],
        degree_nodes: np.ndarray,
        degree_probs: np.ndarray,
        rng: np.random.Generator,
    ) -> set[int]:
        target = int(rng.integers(self.min_nodes, self.max_nodes + 1))
        seed = int(rng.choice(degree_nodes, p=degree_probs))
        selected = {seed}
        queue = deque([seed])
        guard = 0

        while len(selected) < target and guard < target * 30:
            guard += 1
            if not queue:
                queue.append(int(rng.choice(list(selected))))
            src = queue.popleft()
            candidates = [dst for dst in adjacency[src] if dst not in selected]
            rng.shuffle(candidates)
            take = int(rng.integers(1, min(5, len(candidates)) + 1)) if candidates else 0
            for dst in candidates[:take]:
                selected.add(int(dst))
                queue.append(int(dst))
                if len(selected) >= target:
                    break

        return selected

    def _induced_graph(
        self,
        graph_id: int,
        selected: set[int],
        adjacency: dict[int, set[int]],
        id_to_entity: dict[int, str],
        rng: np.random.Generator,
    ) -> Optional[nx.Graph]:
        selected_list = sorted(selected)
        local = {node: idx for idx, node in enumerate(selected_list)}
        G = nx.Graph()
        for node in selected_list:
            G.add_node(local[node], entity=id_to_entity.get(node, str(node)))

        edges = []
        for src in selected_list:
            for dst in adjacency[src]:
                if dst in local and src < dst:
                    edges.append((local[src], local[dst]))

        if len(edges) == 0:
            return None
        G.add_edges_from(edges)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            if len(largest) < self.min_nodes:
                return None
            G = G.subgraph(largest).copy()
            G = nx.convert_node_labels_to_integers(G)

        if G.number_of_edges() > self.max_edges:
            tree_edges = list(nx.minimum_spanning_tree(G).edges())
            tree_set = {tuple(sorted(edge)) for edge in tree_edges}
            extra_edges = [
                edge for edge in G.edges()
                if tuple(sorted(edge)) not in tree_set
            ]
            rng.shuffle(extra_edges)
            keep = tree_edges + extra_edges[:max(0, self.max_edges - len(tree_edges))]
            H = nx.Graph()
            H.add_nodes_from(G.nodes(data=True))
            H.add_edges_from(keep)
            G = H

        G.graph.update(
            dict(
                name=f"{self.dataset_name.lower()}_{graph_id:05d}",
                dataset=self.dataset_name,
            )
        )
        return G

    def generate(self) -> Iterator[nx.Graph]:
        adjacency, id_to_entity, _ = self._read_train_graph()
        degree_nodes = np.array(sorted(adjacency), dtype=np.int64)
        degrees = np.array([len(adjacency[int(node)]) for node in degree_nodes], dtype=np.float64)
        degree_probs = np.power(np.maximum(degrees, 1.0), 0.75)
        degree_probs = degree_probs / degree_probs.sum()

        rng = np.random.default_rng(self.seed)
        made = 0
        attempts = 0
        max_attempts = self.num_graphs * 30
        with tqdm(total=self.num_graphs, desc=f"Sample {self.dataset_name} graphs") as pbar:
            while made < self.num_graphs and attempts < max_attempts:
                attempts += 1
                selected = self._sample_nodes(adjacency, degree_nodes, degree_probs, rng)
                if len(selected) < self.min_nodes:
                    continue
                graph = self._induced_graph(made, selected, adjacency, id_to_entity, rng)
                if graph is None or graph.number_of_nodes() < self.min_nodes:
                    continue
                made += 1
                pbar.update(1)
                yield graph

        if made < self.num_graphs:
            raise RuntimeError(
                f"Only sampled {made}/{self.num_graphs} graphs from {self.dataset_name}."
            )

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
            with open(self.index_path, "w", encoding="utf-8") as index_file:
                index_file.write("\n".join(self.index))

        data_list = map(self.datatype, self.generate())
        data_list = filter_and_save_index(data_list)
        data_list = map(self.pre_transform, data_list)
        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])


class FB15k237SampledDataset(_KGSampledDataset):
    DEFAULT_NAME = "FB15k237Sampled"
    URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"


class WN18RRSampledDataset(_KGSampledDataset):
    DEFAULT_NAME = "WN18RRSampled"
    URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original"
