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

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False) -> 'TwoPlayerGraph()':
        Graph.__init__(self, config_yaml=config_yaml, save_flag=save_flag)
        self._graph_name = graph_name
        self._graph = nx.MultiDiGraph(name=graph_name)

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

        # add nodes
        for node_name, attr in graph_yaml['nodes'].items():

            self.add_state(node_name)

            for attr_name, attr_val in attr.items():

                self.add_state_attribute(node_name, attr_name, attr_val)

            # add init and accepting node attribute
            if node_name == graph_yaml['start_state']:

                self.add_initial_state(node_name)

        # add edges
        for start_name, edge_dict in graph_yaml['edges'].items():
            for end_name, attr in edge_dict.items():

                self.add_edge(start_name,
                              end_name,
                              **attr)

    def fancy_graph(self, color=("lightgrey", "red", "purple"), **kwargs) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            obs = n[1].get('observation')
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
        # return list(nx.simple_cycles(self._graph))
        # return list(nx.strongly_connected_components(self._graph))

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
                 start_agent: str = 'sys',
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

        if from_file:
            graph_yaml = self._from_yaml(config_yaml)

        if minigrid is not None:
            graph_yaml = self._from_minigrid(minigrid, start_agent)

        self._instance.construct_graph(graph_yaml)

        if plot:
            self._instance.plot_graph(view=view, format=format)

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:
        config_data = self.load_YAML_config_data(config_file_name)

        return config_data

    def _from_minigrid(self, minigrid_environment, start_agent) -> dict:
        config_data = minigrid_environment.extract_two_player_game(start_agent)

        # Translate minigrid player to this library's player names
        node_names = list(config_data['nodes'].keys())
        for node_name in node_names:
            minigrid_player = config_data['nodes'][node_name]['player']

            if minigrid_player == 'sys':
                player = 'eve'
            else:
                player = 'adam'

            config_data['nodes'][node_name]['player'] = player

        return config_data
