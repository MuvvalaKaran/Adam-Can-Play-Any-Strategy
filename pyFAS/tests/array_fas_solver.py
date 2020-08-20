# a script to build a Directed graph and test the ArraFAS solver
import networkx as nx

from networkx import DiGraph

# import local packages
from solvers import solver_factory

# abs path to the file
FILE_ADD = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/pyFAS/files/gnm_300_2500_graph_w_extra_100_path_len_0.edges"


def get_digraph(read_from_file: bool = False, scc_wiki: bool = False) -> DiGraph:

    if read_from_file:
        _graph = nx.read_edgelist(FILE_ADD, create_using=nx.DiGraph(), nodetype=int)
        if nx.is_directed_acyclic_graph(_graph):
            print("Grpah already acyclic")
            return None
    elif scc_wiki:
        _graph = nx.DiGraph(name="fas_example")
        _graph.add_nodes_from(range(16))
        # _graph.add_edges_from([(0, 2), (2, 1), (1, 0), (3, 2), (4, 3), (2, 4)])
        # _graph.add_edges_from([(1, 5), (4, 5), (5, 6), (5, 8)])
        # _graph.add_edges_from([(8, 6), (6, 7), (7, 8), (7, 9), (8, 9)])
        # _graph.add_edges_from([(10, 9), (10, 11), (11, 10), (15, 10), (5, 10)])
        _graph.add_edges_from([(12, 13), (13, 15), (15, 14), (14, 12), (13, 14)])
    else:
        _graph = nx.DiGraph(name="fas_example")
        _graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
        _graph.add_edge(0, 1)
        _graph.add_edge(0, 2)
        _graph.add_edge(1, 2)
        _graph.add_edge(2, 3)
        _graph.add_edge(3, 4)
        _graph.add_edge(3, 5)
        _graph.add_edge(3, 6)
        _graph.add_edge(4, 6)
        _graph.add_edge(5, 4)
        _graph.add_edge(5, 7)
        _graph.add_edge(6, 0)
        _graph.add_edge(7, 2)
        _graph.add_edge(7, 1)

    return _graph

if __name__ == "__main__":

    graph = get_digraph(read_from_file=False, scc_wiki=True)
    if graph is not None:
        solver = solver_factory.get("array_fas", graph=graph)
        solver.solve(debug=True)