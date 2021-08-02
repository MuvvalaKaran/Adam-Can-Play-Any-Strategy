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
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

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

    @classmethod
    def build_running_ex(cls: 'TwoPlayerGraph',
                         graph_name: str,
                         config_yaml: str,
                         save_flag: bool = False) -> 'TwoPlayerGraph()':
        """
        A class method that constructs the sample three state graph for you
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :return: An concrete instance of the built three state graph as described in the configuration above
        """

        # define constant weights
        lambda_const = +0
        const_a = 20

        nstate_graph = TwoPlayerGraph(graph_name=graph_name, config_yaml=config_yaml, save_flag=save_flag)
        nstate_graph.construct_graph()

        # nstate_graph.add_weighted_edges_from([('(v1, 1)', '(v2, 1)', -2),
        #                                       ('(v2, 1)', '(v4, 1)', 0),
        #                                       ('(v1, 1)', '(v3, 1)', -1),
        #                                       ('(v3, 1)', '(v5, 1)', 0),
        #                                       # ('(v5, 1)', '(v3, 1)', -1),
        #                                       ('(v5, 1)', 'trap', 0),
        #                                       # ('(v5, 1)', '(v2, 1)', -2),
        #                                       ('(v1, 0)', '(v2, 0)', -2),
        #                                       ('(v1, 0)', '(v3, 0)', -1),
        #                                       ('(v2, 0)', '(v4, 0)', 0),
        #                                       ('(v3, 0)', '(v5, 0)', 0),
        #                                       # ('(v5, 0)', '(v2, 0)', -2),
        #                                       # ('(v5, 0)', '(v3, 0)', -1)]),
        #                                       ('(v5, 0)', 'trap', 0)])
        #
        # # add human interventions
        # nstate_graph.add_weighted_edges_from([('(v2, 1)', '(v1, 0)', 0),
        #                                       ('(v2, 1)', '(v4, 0)', 0),
        #                                       ('(v2, 1)', '(v5, 0)', 0),
        #                                       ('(v3, 1)', '(v1, 0)', 0),
        #                                       ('(v3, 1)', '(v4, 0)', 0),
        #                                       ('(v3, 1)', '(v5, 0)', 0)])

        # nstate_graph.add_weighted_edges_from([('(v1, 1)', '(v2, 1)', const_a),
        #                                       ('(v2, 1)', '(v9, 1)', lambda_const),
        #                                       ('(v1, 1)', '(v3, 1)', 1),
        #                                       ('(v3, 1)', '(v6, 1)', lambda_const),
        #                                       ('(v6, 1)', '(v5, 1)', 1),
        #                                       ('(v6, 1)', '(v7, 1)', 1),
        #                                       ('(v7, 1)', '(v8, 1)', lambda_const),
        #                                       ('(v5, 1)', '(v4, 1)', lambda_const),
        #                                       ('(v4, 1)', '(v2, 1)', 1),  # adding auxiliary human intervention edges
        #                                       ('(v2, 1)', '(v1, 0)', 0),
        #                                       ('(v2, 1)', '(v9, 0)', 0),
        #                                       ('(v2, 1)', '(v4, 0)', 0),
        #                                       ('(v2, 1)', '(v6, 0)', 0),
        #                                       ('(v2, 1)', '(v8, 0)', 0),
        #                                       ('(v3, 1)', '(v1, 0)', 0),
        #                                       ('(v3, 1)', '(v9, 0)', 0),
        #                                       ('(v3, 1)', '(v4, 0)', 0),
        #                                       ('(v3, 1)', '(v6, 0)', 0),
        #                                       ('(v3, 1)', '(v8, 0)', 0),
        #                                       ('(v7, 1)', '(v1, 0)', 0),
        #                                       ('(v7, 1)', '(v9, 0)', 0),
        #                                       ('(v7, 1)', '(v4, 0)', 0),
        #                                       ('(v7, 1)', '(v6, 0)', 0),
        #                                       ('(v7, 1)', '(v8, 0)', 0),
        #                                       ('(v5, 1)', '(v1, 0)', 0),
        #                                       ('(v5, 1)', '(v9, 0)', 0),
        #                                       ('(v5, 1)', '(v4, 0)', 0),
        #                                       ('(v5, 1)', '(v6, 0)', 0),
        #                                       ('(v5, 1)', '(v8, 0)', 0),  # add edges after human has intervened once
        #                                       ('(v1, 0)', '(v2, 0)', const_a),
        #                                       ('(v2, 0)', '(v9, 0)', 0),
        #                                       ('(v1, 0)', '(v3, 0)', 1),
        #                                       ('(v3, 0)', '(v6, 0)', 0),
        #                                       ('(v6, 0)', '(v5, 0)', 1),
        #                                       ('(v6, 0)', '(v7, 0)', 1),
        #                                       ('(v7, 0)', '(v8, 0)', 0),
        #                                       ('(v5, 0)', '(v4, 0)', 0),
        #                                       ('(v4, 0)', '(v2, 0)', 1)])

        # lets add a transition to accepting state form v4 and a transition to trap state from v5
        # nstate_graph.add_edge('(v9, 0)', 'accept_all', weight=0)
        # nstate_graph.add_edge('(v9, 1)', 'accept_all', weight=0)
        # nstate_graph.add_edge('accept_all', 'accept_all', weight=0)
        # nstate_graph.add_state_attribute('accept_all', 'player', 'eve')
        # nstate_graph.add_accepting_state('accept_all')

        # adding trap state
        # nstate_graph.add_edge('trap', 'trap', weight=-1*math.inf)
        # nstate_graph.add_edge('trap', 'trap', weight=-3)
        # nstate_graph.add_state_attribute('trap', 'player', 'adam')

        # nstate_graph.add_edge('v5', 'trap', weight=0)
        # nstate_graph.add_edge('(v8, 1)', 'trap', weight=0)
        # nstate_graph.add_edge('(v8, 0)', 'trap', weight=0)
        # nstate_graph.add_edge('trap', 'trap', weight=0)
        # nstate_graph.add_state_attribute('trap', 'player', 'eve')

        # nstate_graph.add_state_attribute('(v1, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v2, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v3, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v4, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v5, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v1, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v2, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v3, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v4, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v5, 0)', 'player', 'eve')

        # nstate_graph.add_state_attribute('(v1, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v2, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v3, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v4, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v5, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v6, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v7, 1)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v8, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v9, 1)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v1, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v2, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v3, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v4, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v5, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v6, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v7, 0)', 'player', 'adam')
        # nstate_graph.add_state_attribute('(v8, 0)', 'player', 'eve')
        # nstate_graph.add_state_attribute('(v9, 0)', 'player', 'eve')
        #
        # nstate_graph.add_initial_state('(v1, 1)')

        ###############################################################

        nstate_graph.add_weighted_edges_from([('v1', 'v2', 1),
                                              ('v2', 'v4', 0),
                                              ('v2', 'v3', 0),
                                              ('v3', 'v4', 1),
                                              ('v3', 'v6', 1),
                                              ('v4', 'v7', 0),
                                              ('v4', 'v5', 0),
                                              ('v5', 'accept_all', 0),
                                              ('v6', 'v7', 0),
                                              ('v6', 'v5', 0),
                                              ('v7', 'trap', 0),
                                              ('trap', 'trap', 0),
                                              ('accept_all', 'accept_all', 0)])

        nstate_graph.add_state_attribute('v1', 'player', 'eve')
        nstate_graph.add_state_attribute('v2', 'player', 'adam')
        nstate_graph.add_state_attribute('v3', 'player', 'eve')
        nstate_graph.add_state_attribute('v4', 'player', 'adam')
        nstate_graph.add_state_attribute('v5', 'player', 'eve')
        nstate_graph.add_state_attribute('v6', 'player', 'adam')
        nstate_graph.add_state_attribute('v7', 'player', 'eve')
        nstate_graph.add_state_attribute('accept_all', 'player', 'eve')
        nstate_graph.add_state_attribute('trap', 'player', 'eve')

        nstate_graph.add_initial_state('v1')
        nstate_graph.add_accepting_state('accept_all')

        # # for add the constant to each edge
        # for _e in nstate_graph._graph.edges():
        #     _u = _e[0]
        #     _v = _e[1]
        #
        #     _w = nstate_graph.get_edge_weight(_u, _v)
        #     _new_w = _w + lambda_const
        #
        #     nstate_graph._graph[_u][_v][0]['weight'] = _new_w

        # nstate_graph.add_weighted_edges_from([('v1', 'v2', -2),
        #                                       ('v2', 'v1', 0),
        #                                       ('v2', 'v4', 0),
        #                                       ('v4', 'v2', -2),
        #                                       ('v1', 'v3', -1),
        #                                       ('v3', 'v5', 0),
        #                                       ('v5', 'v3', -1)])
        #
        # nstate_graph.add_state_attribute('v1', 'player', 'eve')
        # nstate_graph.add_state_attribute('v2', 'player', 'adam')
        # nstate_graph.add_state_attribute('v3', 'player', 'adam')
        # nstate_graph.add_state_attribute('v4', 'player', 'eve')
        # nstate_graph.add_state_attribute('v5', 'player', 'eve')
        #
        # # adding edges to the accepting state
        # nstate_graph.add_edge('v4', 'accept_all', weight=0)
        # nstate_graph.add_edge('accept_all', 'accept_all', weight=0)
        # nstate_graph.add_state_attribute('accept_all', 'player', 'eve')

        # add the trap state
        # nstate_graph.add_edge('v5', 'trap', weight=0)
        # nstate_graph.add_edge('trap', 'trap', weight=-math.inf)
        # nstate_graph.add_state_attribute('trap', 'player', 'adam')
        #
        # nstate_graph.add_initial_state('v1')
        # nstate_graph.add_accepting_state('accept_all')

        return nstate_graph

    @classmethod
    def build_twa_example(cls: 'TwoPlayerGraph',
                          graph_name: str,
                          config_yaml: str,
                          save_flag: bool = False) -> 'TwoPlayerGraph()':

        """
        A class method to construct a sample Two player target weighted arena (TWA)

        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :return: An concrete instance of the built Target weighted graph
        """

        _twa_graph: TwoPlayerGraph = TwoPlayerGraph(graph_name=graph_name, config_yaml=config_yaml, save_flag=save_flag)
        _twa_graph.construct_graph()

        _twa_graph.add_state('A', player="adam")
        _twa_graph.add_state('C', player="eve")
        _twa_graph.add_state('B', player="eve")
        _twa_graph.add_state('D', player="adam")
        _twa_graph.add_state('E', player="eve")
        _twa_graph.add_state('F', player="adam")
        _twa_graph.add_state('G', player="eve")
        _twa_graph.add_state('H', player="eve")
        _twa_graph.add_state('I', player="eve")
        _twa_graph.add_state('J', player="eve")

        _twa_graph.add_weighted_edges_from([('A', 'B', 0),
                                            ('A', 'C', 0),
                                            ('B', 'C', 0),
                                            ('B', 'D', 0),
                                            ('C', 'F', 0),
                                            ('C', 'E', 3),
                                            ('D', 'G', 3),
                                            ('D', 'H', 0),
                                            ('F', 'I', 4),
                                            ('F', 'J', 0),
                                            ('F', 'B', 0)])
        _twa_graph.add_initial_state('A')
        _twa_graph.add_accepting_states_from(['E', 'G', 'H', 'I', 'J'])

        # for leaf nodes let's add a self-loop (for the construction to be in consistent with infinite play game)
        for _s in _twa_graph._graph.nodes():
            if len(list(_twa_graph._graph.successors(_s))) == 0:
                _twa_graph.add_weighted_edges_from([(_s, _s, 0)])

        return _twa_graph

    @classmethod
    def build_ewa_example(cls: 'TwoPlayerGraph',
                          graph_name: str,
                          config_yaml: str,
                          save_flag: bool = False) -> 'TwoPlayerGraph()':

        """
        A class method to construct a sample Two player Edge weighted arena (WWA)

        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :return: An concrete instance of the built Earget weighted graph
        """

        _ewa_graph: TwoPlayerGraph = TwoPlayerGraph(graph_name=graph_name, config_yaml=config_yaml, save_flag=save_flag)
        _ewa_graph.construct_graph()

        # example from the paper
        # _ewa_graph.add_state('A', player="adam")
        # _ewa_graph.add_state('C', player="eve")
        # _ewa_graph.add_state('B', player="eve")
        # _ewa_graph.add_state('D', player="adam")
        # _ewa_graph.add_state('E', player="eve")
        # _ewa_graph.add_state('F', player="adam")
        # _ewa_graph.add_state('G', player="eve")
        # _ewa_graph.add_state('H', player="eve")
        # _ewa_graph.add_state('I', player="eve")
        # _ewa_graph.add_state('J', player="eve")
        #
        # _ewa_graph.add_weighted_edges_from([('A', 'B', 0),
        #                                     ('A', 'C', 0),
        #                                     ('B', 'C', 1),
        #                                     # ('B', 'D', 1),
        #                                     ('C', 'F', 0),
        #                                     ('C', 'E', 2),
        #                                     ('D', 'G', 2),
        #                                     ('D', 'H', 0),
        #                                     ('F', 'B', 0),
        #                                     ('F', 'I', 2),
        #                                     ('F', 'J', 0)])
        # _ewa_graph.add_initial_state('A')
        # _ewa_graph.add_accepting_states_from(['E', 'G', 'H', 'I', 'J'])

        # simple example
        # _ewa_graph.add_state('A', player="adam")
        # _ewa_graph.add_state('B', player="eve")
        # _ewa_graph.add_state('C', player="eve")
        # _ewa_graph.add_state('D', player="adam")
        # _ewa_graph.add_state('E', player="eve")
        # _ewa_graph.add_state('F', player="eve")
        # _ewa_graph.add_state('G', player="eve")
        # _ewa_graph.add_state('H', player="eve")
        #
        # _ewa_graph.add_weighted_edges_from([('A', 'C', 0),
        #                                     ('A', 'B', 0),
        #                                     ('B', 'C', 1),
        #                                     ('B', 'D', 0),
        #                                     ('C', 'E', 2),
        #                                     ('C', 'F', 1),
        #                                     ('F', 'B', 0),
        #                                     ('D', 'G', 2),
        #                                     ('D', 'H', 0)])
        # _ewa_graph.add_initial_state('A')
        # _ewa_graph.add_accepting_states_from(['E', 'F', 'G'])

        # simple example where init state belongs to Sys player's winning region
        _ewa_graph.add_state('v1', player="eve")
        _ewa_graph.add_state('v2', player="adam")
        _ewa_graph.add_state('v3', player="eve")
        _ewa_graph.add_state('v4', player="adam")
        _ewa_graph.add_state('v5', player="eve")
        _ewa_graph.add_state('v6', player="adam")
        # _ewa_graph.add_state('v7', player="adam")
        # _ewa_graph.add_state('v8', player="eve")
        _ewa_graph.add_state('v9', player="eve")
        # _ewa_graph.add_state('v10', player="eve")
        # _ewa_graph.add_state('v11', player="eve")
        _ewa_graph.add_state('v12', player="eve")

        _ewa_graph.add_weighted_edges_from([('v1', 'v2', 1),
                                            # ('v1', 'v4', 1),
                                            ('v4', 'v5', 0),
                                            ('v2', 'v3', 0),
                                            ('v2', 'v9', 0),
                                            ('v3', 'v4', 1),
                                            ('v3', 'v2', 1),
                                            # ('v5', 'v4', 1),
                                            ('v3', 'v6', 6),
                                            # ('v3', 'v7', 1),
                                            # ('v7', 'v8', 0),
                                            ('v6', 'v9', 0),
                                            # ('v6', 'v4', 5),
                                            ('v5', 'v6', 3),
                                            # ('v8', 'v12', 5),
                                            # ('v10', 'v12', 5),
                                            ('v9', 'v12', 0),
                                            # ('v11', 'v12', 0),
                                            ('v4', 'v9', 0), ('v5', 'v4', 1)])

        _ewa_graph.add_initial_state('v1')
        _ewa_graph.add_accepting_states_from(['v12'])

        # game in Env's winning region
        # _ewa_graph.add_state('v1', player="adam")
        # _ewa_graph.add_state('v2', player="eve")
        # _ewa_graph.add_state('v3', player="adam")
        # _ewa_graph.add_state('v4', player="eve")
        # _ewa_graph.add_state('v5', player="adam")
        # _ewa_graph.add_state('v6', player="adam")
        # _ewa_graph.add_state('v7', player="eve")
        # _ewa_graph.add_state('v8', player="adam")
        # _ewa_graph.add_state('v12', player="adam")
        #
        # _ewa_graph.add_weighted_edges_from([('v1', 'v2', 1),
        #                                     ('v2', 'v3', 0),
        #                                     # ('v2', 'v6', 0),
        #                                     ('v3', 'v4', 1),
        #                                     ('v4', 'v5', 0),
        #                                     ('v3', 'v7', 1),
        #                                     ('v7', 'v8', 0),
        #                                     ('v5', 'v2', 1),
        #                                     ('v8', 'v12', 0), ('v2', 'v6', 2)])#, ('v6', 'v5', 10)])
        #
        # _ewa_graph.add_initial_state('v1')
        # _ewa_graph.add_accepting_states_from(['v12', 'v6'])

        # for leaf nodes let's add a self-loop (for the construction to be in consistent with infinite play game)
        for _s in _ewa_graph._graph.nodes():
            if len(list(_ewa_graph._graph.successors(_s))) == 0:
                _ewa_graph.add_weighted_edges_from([(_s, _s, 0)])

        return _ewa_graph

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

    def export_files_to_prism(self) -> str:
        """
        Export Model and Property files for using with PRISM
        """
        model_filename= self.export_prism_model()
        props_filename = self.export_prism_property()

        return [model_filename, props_filename]

    def export_prism_model(self) -> str:
        """
        Export TwoPlayerGame as a PRISM model to a file
        """
        graph_name = self._graph_name_in_prism()

        # If filepath not given, then use graph_name as a filename
        directory = self._prism_directory()
        filepath = os.path.join(directory, graph_name + '.prism')

        # Create directory if not exists
        file_dir, _ = os.path.split(filepath)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # Info about the graph
        players = self.players
        num_weight = len(self.weight_types)

        num_node = len(self._graph.nodes())
        if num_node < 2:
            warnings.warn('num_node should be greater or equal to 2')
        # Mapping from node name to int val.
        node_int_dict = {n: i for i, n in enumerate(self._graph.nodes())}

        with open(filepath, 'w+') as f:
            # Stochastic Multi-Player Game
            f.write('smg\n\n')

            # TODO: Retrieve actions per player
            actions_per_player = defaultdict(lambda: set())
            for u_node_data in self._graph.nodes.data():
                u_node = u_node_data[0]
                player = u_node_data[1]['player']
                for v_node in self._graph.successors(u_node):
                    action = self.get_edge_attributes(u_node, v_node, 'actions')
                    actions_per_player[player].add(action)

            for i, (player, actions) in enumerate(actions_per_player.items()):
                add_bracket = lambda s: f'[{s}]'
                action_list_str = ', '.join(map(add_bracket, actions))
                f.write(f'player p{i+1}\n')
                f.write(f'\t{action_list_str}\n')
                f.write(f'endplayer\n\n')

            # Game
            f.write(f'module {graph_name}\n')
            f.write(f'\tx : [0..{num_node-1}] init 0;\n') # TODO: What if num_node is 0 or 1?
            for edge in self._graph.edges.data():
                u_node_int = node_int_dict[edge[0]]
                v_node_int = node_int_dict[edge[1]]
                action = edge[2].get('actions')
                # TODO: nondeterministic transitions
                f.write(f"\t[{action}] x={u_node_int} -> 1 : (x'={v_node_int});\n")
            f.write(f'endmodule\n\n')

            # Weight Objective
            for i_weight in range(num_weight):
                weight_name = self.weight_types[i_weight]
                f.write(f'rewards "{weight_name}"\n')
                for edge in self._graph.edges.data():
                    u_node_int = node_int_dict[edge[0]]
                    action = edge[2].get('actions')
                    weight = edge[2].get('weights')[weight_name]
                    f.write(f'\t[{action}] x={u_node_int} : {weight};\n')
                f.write(f'endrewards\n\n')

            # Reachability Objective
            # Set weight of 1 to edges that lead to the accepting state
            # TODO: If there exists a few accepting states, then create a virtual node
            f.write(f'rewards "reach"\n')
            for v_node in self.get_accepting_states():
                for u_node in self._graph.predecessors(v_node):
                    action = self.get_edge_attributes(u_node, v_node, 'actions')
                    u_node_int = node_int_dict[u_node]
                f.write(f'\t[{action}] x={u_node_int} : 1;\n')
            f.write(f'endrewards\n\n')

        return os.path.abspath(filepath)

    def export_prism_property(self) -> str:
        """
        Export Multi-Objective Optimization property to a file
        """
        graph_name = self._graph_name_in_prism()

        # If filepath not given, then use graph_name as a filename
        directory = self._prism_directory()
        filepath = os.path.join(directory, graph_name + '.props')

        # Create directory if not exists
        file_dir, _ = os.path.split(filepath)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # TODO: write property to a file

        # Minimizes weights and maximize reachability
        with open(filepath, 'w+') as f:
            f.write(f'const double r = {0.98};\n')
            for i, name in enumerate(self.weight_types):
                max_weight_value = self._get_max_weight_value(name)
                f.write(f'const double v{i} = {max_weight_value};\n')
            f.write('\n')
            weight_obj_str = [f'R{{"{name}"}}<=v{i}[C]' for i, name in enumerate(self.weight_types)]
            obj_str = ' & '.join([f'R{{"reach"}}>=r[C]'] + weight_obj_str)
            f.write(f'<<p1>> ({obj_str})\n\n')

        return os.path.abspath(filepath)

    def _get_max_weight_value(self, weight_name: str) -> float:
        """
        Compute the maximum possible value for the given weight
        """
        # TODO MAX_WEIGHT_VALUE = A_MAX_WEIGHT * MAX_DEPTH * MAX_WIDTH
        weights = [e[2]['weights'].get(weight_name) for e in self._graph.edges.data()]
        max_single_weight = max(weights)
        max_depth = len(self._graph.nodes())
        successors = [list(self._graph.successors(n)) for n in self._graph.nodes()]
        max_width = max([len(s) for s in successors])

        return max_single_weight * max_depth * max_width

    def _graph_name_in_prism(self):
        """
        Transform graph_name into the prism form
        """
        return self._graph_name.replace(' ', '')

    def _prism_directory(self):
        """
        Get prism's directory
        """
        return os.path.join(self._get_project_root_directory(), 'prism')

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
