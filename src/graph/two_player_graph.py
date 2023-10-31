import networkx as nx
import os
import math
import time
import queue
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Union

# local packages
from .base import Graph
from ..factory.builder import Builder

from graphviz import Digraph


class TwoPlayerGraph(Graph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False,
                 finite: bool = True) -> 'TwoPlayerGraph()':
        Graph.__init__(self, config_yaml=config_yaml, save_flag=save_flag)
        self._graph_name = graph_name
        self._graph = nx.MultiDiGraph(name=graph_name)
        self._finite = finite

    @property
    def graph_name(self):
        return self._graph_name

    @graph_name.setter
    def graph_name(self, graph_name: str):
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
            if edge[2] is None:
                continue
            for weight_type in edge[2].keys():
                weight_types.add(weight_type)
        return list(weight_types)

    def construct_graph(self, graph_yaml: Dict):

        self._graph_yaml = graph_yaml

        if isinstance(graph_yaml['nodes'], list):
            nodes = defaultdict(lambda: {})
            for (node_name, attr) in graph_yaml['nodes']:
                nodes[node_name] = attr
        else:
            nodes = graph_yaml['nodes']

        # add nodes
        for node_name, attr in nodes.items():

            self.add_state(node_name)

            for attr_name, attr_val in attr.items():

                self.add_state_attribute(node_name, attr_name, attr_val)

            # add init and accepting node attribute
            if node_name == graph_yaml['start_state']:

                self.add_initial_state(node_name)

        if isinstance(graph_yaml['edges'], list):
            edges = defaultdict(lambda: {})
            for u, v, attr in graph_yaml['edges']:
                edges[u][v] = attr
        else:
            edges = graph_yaml['edges']


        # add edges
        for start_name, edge_dict in edges.items():
            for end_name, attr in edge_dict.items():
                # Identify if it has multiple edges betw. the start and end node
                are_all_keys_integers = all([isinstance(a, int) for a in attr.keys()])
                has_multiple_edges = are_all_keys_integers

                if has_multiple_edges:
                    self._graph.add_edges_from(
                        [(start_name, end_name, a) for a in attr.values()])
                else:
                    self.add_edge(start_name,
                                end_name,
                                **attr)

    def fancy_graph(self, color=("lightgrey", "red", "purple"),
        start_node: str=None, n_neighbor: int=3,**kwargs) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        node_names = [n[0] for n in nodes]

        # If start_node is given, plot a partial graph
        nodes_to_plot = None
        edges_to_plot = None
        if start_node is not None and start_node not in node_names:
            search_queue = queue.Queue()
            search_queue.put((0, start_node))
            nodes_to_plot = []
            edges_to_plot = []
            while not search_queue.empty():
                ith, u_node = search_queue.get()
                if ith == n_neighbor:
                    continue
                for v_node in self._game._graph.successors(u_node):
                    if v_node not in nodes_to_plot:
                        nodes_to_plot.append(v_node)
                        edges_to_plot.append((u_node, v_node))
                        search_queue.put((ith+1, v_node))

        for n in nodes:
            if nodes_to_plot is not None and n[0] not in nodes_to_plot:
                continue

            obs = n[1].get('ap', [])
            if len(obs) == 0:
                obs = ''
            else:
                obs = str(obs) #"\n".join(obs)
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "xlabel": obs,
                                             "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1], "xlabel": obs})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2], "xlabel": obs})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "circle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            if edges_to_plot is not None and (edge[0], edge[1]) not in edges_to_plot:
                continue

            weight = edge[2].get('weight')
            weight_label = '' if weight is None else str(weight)
            label = str(edge[2].get('actions', '')) + weight_label
            if edge[2].get('strategy') is True:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')), _attributes={'color': 'red'})
                dot.edge(str(edge[0]), str(edge[1]), label=label, _attributes={'color': 'red'})
            else:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')))
                dot.edge(str(edge[0]), str(edge[1]), label=label)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, **kwargs)

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

    def delete_selfloops(self):
        """
        Find cycles (FAS and self loops) and delete them from G

        :args winning_region:   The winning region
        """
        redirected = False

        # For each self loop
        for node in self.get_selfloops():
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
                redirected = True

        if redirected:
            self.add_state('Absorbing', player='eve')

    def delete_cycles(self, winning_region: List):
        """
        Find cycles (FAS and self loops) and delete them from G

        :args winning_region:   The winning region
        """
        cycles = self.get_cycles()

        # For each sccss,
        for cycle in cycles:
            if 'Accepting' in cycle:
                continue

            # TODO: Check if it really is a cycle lol
            if not self._identify_if_really_cycle(cycle):
                continue

            # Check if all nodes in cycle belong to env
            all_env_node = all(['adam'==self.get_state_w_attribute(n, 'player') for n in cycle])
            # Redirect all nodes to absorbing state
            if all_env_node:
                for u_node in cycle:
                    for v_node in self._graph.successors(u_node):
                        if v_node in cycle:
                            edge_attributes = self._graph[u_node][v_node][0]
                            self.add_edge(u_node, 'Absorbing', **edge_attributes)
                            self._graph.remove_edge(u_node, v_node)
                            redirected = True

            # For sys node in cycle
            for u_node in cycle:
                player = self.get_state_w_attribute(u_node, 'player')
                if player == 'eve':
                    # check if it has an edge to one of the nodes in the cycle, then delete it
                    successors = list(self._graph.successors(u_node))
                    for v_node in successors:
                        if v_node in cycle:
                            self._graph.remove_edge(u_node, v_node)

    def get_cycles(self) -> List:
        """
        Find a list of Strongly Connected Components in G

        :return fas:
        """
        cycles_ = list(nx.find_cycle(self._graph))
        return cycles_
        cycles = []
        for cycle in cycles_:
            cycles.append([cycle[0], cycle[1]])
        return cycles

    def get_selfloops(self) -> List:
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

    def _identify_if_really_cycle(self, cycle):
        u_node = cycle[0]

        search_queue = queue.Queue()
        search_queue.put(u_node)
        visited = {node: False for node in cycle}

        while not search_queue.empty(): # Or queue to be empty
            u_node = search_queue.get()

            for v_node in self._graph.successors(u_node):

                #  At least one successor should be in cycle to make it a cycle
                if v_node in cycle and not visited[v_node]:
                    visited[v_node] = True
                    search_queue.put(v_node)

        return all(list(visited.values()))


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
                 minigrid = None,
                 n_step: int = None,
                 save_flag: bool = False,
                 from_file: bool = False,
                 pre_built: bool = False,   # TODO: Delete
                 plot: bool = False,
                 view: bool = True,
                 format: str = 'pdf') -> TwoPlayerGraph:
        """
        Return an initialized TwoPlayerGraph instance given the configuration data
        :param graph_name : Name of the graph
        :return: A concrete/active instance of the TwoPlayerGraph
        """
        self._instance = TwoPlayerGraph(graph_name, config_yaml, save_flag)

        graph_yaml = None
        if from_file:
            graph_yaml = self._from_yaml(config_yaml)

        if graph_yaml is None and minigrid is not None:
            graph_yaml = self._from_minigrid(minigrid, n_step)

        if graph_yaml:
            self._instance.construct_graph(graph_yaml)

        if plot:
            self._instance.plot_graph(view=view, format=format)

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:
        config_data = self.load_YAML_config_data(config_file_name)

        return config_data

    def _from_minigrid(self, minigrid_environment, n_step) -> dict:
        config_data = minigrid_environment.extract_transition_system(n_step)

        # Translate minigrid player to this library's player names
        node_names = list(config_data['nodes'].keys())
        for node_name in node_names:
            minigrid_player = config_data['nodes'][node_name]['player']

            if minigrid_player == 'sys':
                player = 'eve'
            else:
                player = 'adam'

            config_data['nodes'][node_name]['player'] = player
            config_data['nodes'][node_name]['ap'] = \
                config_data['nodes'][node_name]['observation']
            del config_data['nodes'][node_name]['observation']

        return config_data
