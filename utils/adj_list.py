from collections import defaultdict
import networkx as nx
from typing import Literal


class FlightNode:
    def __init__(
        self, dep_port: str, arr_port: str, dep_time: int, arr_time: int, node_id: int
    ):
        self.dep_port = dep_port
        self.arr_port = arr_port
        self.dep_time = dep_time
        self.arr_time = arr_time
        self.node_id = node_id

    def __repr__(self):
        return (
            "{"
            + f"{self.dep_port}->{self.arr_port} [{self.dep_time}, {self.arr_time}]"
            + "}"
        )

    def type(self) -> Literal["maintenance", "flight"]:
        if self.dep_port == self.arr_port:
            return "maintenance"
        else:
            return "flight"


def read_data(path: str) -> list[FlightNode]:
    import pandas as pd

    flight_nodes = []
    data = pd.read_csv(path)
    for row_idx in range(data.shape[0]):
        flight_nodes.append(
            FlightNode(
                data["depPort"][row_idx],
                data["arrPort"][row_idx],
                data["depTime"][row_idx],
                data["arrTime"][row_idx],
                data["series"][row_idx],
            )
        )
    return flight_nodes


class AdjList:
    def __init__(self, flight_nodes: list[FlightNode]) -> None:
        self.start_node, self.end_node, self.adj_list = self.edge_idx_2_adj_list(
            flight_nodes
        )
        self.graph = None

    @property
    def edge_idx(self):
        edge_idx_list = []
        for key in self.adj_list:
            for node in self.adj_list[key]:
                edge_idx_list.append((key, node))
        return edge_idx_list

    @staticmethod
    def edge_idx_2_adj_list(
        flight_nodes: list[FlightNode],
    ) -> tuple[int, int, dict[int, list[int]]]:
        adj_list = defaultdict(list)
        flight_nodes_length = len(flight_nodes)
        start_node = flight_nodes[0].node_id
        end_node = flight_nodes[0].node_id
        for i in range(flight_nodes_length):
            if flight_nodes[i].node_id < start_node:
                start_node = flight_nodes[i].node_id
            if flight_nodes[i].node_id > end_node:
                end_node = flight_nodes[i].node_id
            for j in range(i + 1, flight_nodes_length):
                if (
                    flight_nodes[i].arr_port == flight_nodes[j].dep_port
                    and flight_nodes[i].arr_time <= flight_nodes[j].dep_time
                ):
                    adj_list[flight_nodes[i].node_id].append(flight_nodes[j].node_id)
        return start_node - 1, end_node + 1, adj_list

    def add_start_end(self, start_list: list[int], end_list: list[int]) -> None:
        self.adj_list[self.start_node] = start_list
        self.adj_list[self.end_node] = end_list
        self.graph = nx.DiGraph(self.edge_idx)

    def topological_sort(self) -> list[int]:
        edges = nx.bfs_edges(self.graph, self.start_node)
        return [self.start_node] + list(map(lambda x: x[1], edges)) + [self.end_node]

    @classmethod
    def from_file(cls, csv_path: str):
        flight_nodes = read_data(csv_path)
        return cls(flight_nodes)


if __name__ == "__main__":
    # flights = read_data("ori_data.csv")
    # adj_list = AdjList(flights)
    adj_list = AdjList.from_file("../ori_data.csv")
    adj_list.add_start_end(
        [0, 14, 29, 42, 56, 71, 87, 101, 114, 129, 142, 154],
        [13, 28, 41, 55, 70, 86, 100, 113, 128, 141, 153, 165],
    )
