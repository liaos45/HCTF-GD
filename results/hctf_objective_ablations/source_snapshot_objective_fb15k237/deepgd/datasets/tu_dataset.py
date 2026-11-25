from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
from typing import Iterator, Optional, TypeVar

import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.datasets import TUDataset
from tqdm.auto import tqdm


DATATYPE = TypeVar("DATATYPE", bound=BaseData)


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


class TUGraphDrawingDataset(pyg.data.InMemoryDataset):
    DEFAULT_TU_NAME = "IMDB-BINARY"
    DEFAULT_NAME = "TU_IMDB_BINARY"

    def __init__(
        self,
        *,
        root: str = DATASET_ROOT,
        name: str = DEFAULT_NAME,
        tu_name: str = DEFAULT_TU_NAME,
        index: Optional[list[str]] = None,
        datatype: type[DATATYPE] = GraphDrawingData,
    ):
        self.dataset_name = name
        self.tu_name = tu_name
        self.index = index
        self.datatype = datatype
        self.tu_root = os.path.join(root, "_tu_raw")
        super().__init__(
            root=os.path.join(root, name),
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
        return [f"{self.tu_name}.downloaded"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]

    @property
    def marker_path(self) -> str:
        return os.path.join(self.raw_dir, self.raw_file_names[0])

    def download(self) -> None:
        os.makedirs(self.raw_dir, exist_ok=True)
        TUDataset(root=self.tu_root, name=self.tu_name, use_node_attr=False, use_edge_attr=False)
        with open(self.marker_path, "w", encoding="utf-8") as marker:
            marker.write(self.tu_name)

    def generate(self) -> Iterator[nx.Graph]:
        dataset = TUDataset(
            root=self.tu_root,
            name=self.tu_name,
            use_node_attr=False,
            use_edge_attr=False,
        )
        prefix = self.tu_name.lower().replace("-", "_")
        for graph_id, data in enumerate(tqdm(dataset, desc=f"Loading {self.tu_name} graphs")):
            graph = nx.Graph()
            graph.add_nodes_from(range(int(data.num_nodes)))
            edge_index = data.edge_index.long().cpu().numpy()
            for src, dst in edge_index.T:
                src = int(src)
                dst = int(dst)
                if src != dst:
                    graph.add_edge(src, dst)
            label = int(data.y.item()) if getattr(data, "y", None) is not None else -1
            graph.graph.update(
                dict(
                    name=f"{prefix}_{graph_id:05d}",
                    dataset=self.dataset_name,
                    graph_label=label,
                )
            )
            yield graph

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


class IMDBBinaryDataset(TUGraphDrawingDataset):
    DEFAULT_TU_NAME = "IMDB-BINARY"
    DEFAULT_NAME = "IMDBBinary"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", self.DEFAULT_NAME)
        kwargs.setdefault("tu_name", self.DEFAULT_TU_NAME)
        super().__init__(**kwargs)


class PROTEINSDataset(TUGraphDrawingDataset):
    DEFAULT_TU_NAME = "PROTEINS"
    DEFAULT_NAME = "PROTEINS"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", self.DEFAULT_NAME)
        kwargs.setdefault("tu_name", self.DEFAULT_TU_NAME)
        super().__init__(**kwargs)


class ENZYMESDataset(TUGraphDrawingDataset):
    DEFAULT_TU_NAME = "ENZYMES"
    DEFAULT_NAME = "ENZYMES"

    def __init__(self, **kwargs):
        kwargs.setdefault("name", self.DEFAULT_NAME)
        kwargs.setdefault("tu_name", self.DEFAULT_TU_NAME)
        super().__init__(**kwargs)
