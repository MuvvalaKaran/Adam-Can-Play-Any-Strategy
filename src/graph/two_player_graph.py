import networkx as nx
import os
import math
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Union

# local packages
from .base import Graph
from ..factory.builder import Builder

from graphviz import Digraph
from pyFAS.solvers import solver_factory

class TwoPlayerGraph(Graph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False) -> 'TwoPlayerGraph()':
        Graph.__init__(self, config_yaml=config_yaml, save_flag=save_flag)
        self._graph_name = graph_name

    @property
    def graph_name(self):
        return self._graph_name

    @graph_name.setter
    def graph_name(self, name, prefix: str = None, suffix: str = None):
        if prefix:
            graph_name = prefix + graph_name

        if suffix:
            graph_name = graph_name + suffix

        self._graph_name = graph_name

    @property
    def players(self) -> List[str]:
        return set(self._graph.nodes.data('player'))

    @property
    def weight_types(self) -> List[str]:
        edges = self._graph.edges.data('weights')
        if len(edges)<2:
            return []

        weight_types = set()
        for edge in edges:
            for weight_type in edge[2].keys():
                weight_types.add(weight_type)
        return list(weight_types)

    def construct_graph(self):
        two_player_graph: nx.MultiDiGraph = nx.MultiDiGraph(name=self._graph_name)
        # add this graph object of type of Networkx to our Graph class
        self._graph = two_player_graph

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            ap = n[1].get('ap')
            ap = "{" + str(ap) + "}"
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "xlabel": ap,
                                             "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1], "xlabel": ap})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2], "xlabel": ap})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "circle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            if edge[2].get('strategy') is True:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')), _attributes={'color': 'red'})
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')), _attributes={'color': 'red'})
            else:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')))
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')))

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, True)

    def print_edges(self):
        print("=====================================")
        print(f"Printing {self._graph_name} edges \n")
        super().print_edges()
        print("=====================================")

    def print_nodes(self):
        print("=====================================")
        print(f"Printing {self._graph_name} nodes \n")
        super().print_nodes()
        print("=====================================")

    def get_max_weight(self) -> float:
        max_weight: int = 0
        # loop through all the edges and return the max weight
        for _e in self._graph.edges.data("weight"):
            if abs(_e[2]) != math.inf and abs(_e[2]) > abs(max_weight):
                max_weight = _e[2]

        return max_weight

    def _get_set_ap(self) -> set:
        """
        A helper method that return a set of observations associated with each state in the transition system
        :return:
        """

        atomic_propositions: set = set()
        for _n in self._graph.nodes.data():
            atomic_propositions.add(_n[1].get('ap'))

        return atomic_propositions

    def _get_next_node(self, curr_node, symbol) -> Tuple:
        # TODO
        pass

    def trace_cumulative_cost(self, actions: List):
        # TODO
        pass

    def strategy_cumulative_cost(self, strategy: Dict):
        # TODO
        pass

    def delete_cycles(self, winning_region: List):
        """
        Find cycles (FAS and self loops) and delete them from G

        :args winning_region:   The winning region
        """
        self.add_state('Absorbing', player='eve')
        # For each self loop
        for node in self.identify_loops():
            if node == 'Accepting':
                continue
            # check if the loop belongs to sys or env
            player = self.get_state_w_attribute(node, 'player')
            # If sys, delete the edge
            if player == 'sys':
                self._graph.remove_edge(node, node)
            # If env, redirect the edge to an absorbing state
            else:
                edge_attributes = self._graph[node][node][0]
                self.add_edge(node, 'Absorbing', **edge_attributes)
                self._graph.remove_edge(node, node)

        sccs = self.identify_sccs()
        # For each sccss,
        for scc in sccs:
            # Check if all nodes in scc belong to env
            all_env_node = all(['adam'==self.get_state_w_attribute(n, 'player') for n in scc])
            # Redirect all nodes to absorbing state
            if all_env_node:
                for u_node in scc:
                    for v_node in self._graph.successors(u_node):
                        if v_node in scc:
                            edge_attributes = self._graph[u_node][v_node][0]
                            self.add_edge(u_node, 'Absorbing', **edge_attributes)
                            self._graph.remove_edge(u_node, v_node)

            # For sys node in scc
            for u_node in scc:
                player = self.get_state_w_attribute(u_node, 'player')
                if player == 'eve':
                    # check if it has an edge to one of the nodes in the scc, then delete it
                    successors = list(self._graph.successors(u_node))
                    for v_node in successors:
                        if v_node in scc:
                            self._graph.remove_edge(u_node, v_node)

    def identify_sccs(self) -> List:
        """
        Find a list of Strongly Connected Components in G

        :return fas:
        """
        graph = self._graph
        solver = solver_factory.get("array_fas", graph=graph)
        solver.solve(debug=True)

        return solver.get_fas_set()

    def identify_loops(self) -> List:
        """
        Find self loops in G
        """
        graph = self._graph

        loops = []
        for u_node in graph.nodes():
            for v_node in graph.successors(u_node):
                if u_node == v_node:
                    loops.append(u_node)

        return loops

class TwoPlayerGraphBuilder(Builder):
    """
    Implements the generic graph builder class for TwoPlayerGraph
    """

    def __init__(self) -> 'TwoPlayerGraphBuilder()':
        """
        Constructs a new instance of the TwoPlayerGraph Builder

        attr: pre_built : instance variable indicating if the user wants to build his own graph or use the internal one
        attr: two_player_graphs : a dictionary that has pre built instances of the TwoPlayerGraph key to the graph key
        """

        Builder.__init__(self)

    def __call__(self,
                 graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 from_file: bool = False,
                 pre_built: bool = False,
                 plot: bool = False) -> TwoPlayerGraph:
        """
        Return an initialized TwoPlayerGraph instance given the configuration data
        :param graph_name : Name of the graph
        :return: A concrete/active instance of the TwoPlayerGraph
        """
        self._instance = TwoPlayerGraph(graph_name, config_yaml, save_flag)
        self._instance.construct_graph()

        if from_file:
            self._instance._graph_yaml = self._from_yaml(config_yaml)

        if pre_built:
            if graph_name == "two_player_graph":
                self._instance = TwoPlayerGraph.build_running_ex(graph_name, config_yaml, save_flag)
            elif graph_name == "target_weighted_arena":
                self._instance = TwoPlayerGraph.build_twa_example(graph_name, config_yaml, save_flag)
            elif graph_name == "edge_weighted_arena":
                self._instance = TwoPlayerGraph.build_ewa_example(graph_name, config_yaml, save_flag)
            else:
                warnings.warn("Please enter a valid graph name to load a pre-built graph.")

        if plot:
            self._instance.plot_graph()

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:
        config_data = self.load_YAML_config_data(config_file_name)

        return config_data
