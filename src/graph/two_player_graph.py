import networkx as nx
import math

# local packages
from .base import Graph
from ..factory.builder import Builder

from graphviz import Digraph


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
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')), _attributes={'color': 'red'})
            else:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')))

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
        :return: An concrete instance of the built three state grpah as described in the configuration above
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

        nstate_graph.add_weighted_edges_from([('(v1, 1)', '(v2, 1)', const_a),
                                              ('(v2, 1)', '(v9, 1)', lambda_const),
                                              ('(v1, 1)', '(v3, 1)', 1),
                                              ('(v3, 1)', '(v6, 1)', lambda_const),
                                              ('(v6, 1)', '(v5, 1)', 1),
                                              ('(v6, 1)', '(v7, 1)', 1),
                                              ('(v7, 1)', '(v8, 1)', lambda_const),
                                              ('(v5, 1)', '(v4, 1)', lambda_const),
                                              ('(v4, 1)', '(v2, 1)', 1),  # adding auxiliary human intervention edges
                                              ('(v2, 1)', '(v1, 0)', 0),
                                              ('(v2, 1)', '(v9, 0)', 0),
                                              ('(v2, 1)', '(v4, 0)', 0),
                                              ('(v2, 1)', '(v6, 0)', 0),
                                              ('(v2, 1)', '(v8, 0)', 0),
                                              ('(v3, 1)', '(v1, 0)', 0),
                                              ('(v3, 1)', '(v9, 0)', 0),
                                              ('(v3, 1)', '(v4, 0)', 0),
                                              ('(v3, 1)', '(v6, 0)', 0),
                                              ('(v3, 1)', '(v8, 0)', 0),
                                              ('(v7, 1)', '(v1, 0)', 0),
                                              ('(v7, 1)', '(v9, 0)', 0),
                                              ('(v7, 1)', '(v4, 0)', 0),
                                              ('(v7, 1)', '(v6, 0)', 0),
                                              ('(v7, 1)', '(v8, 0)', 0),
                                              ('(v5, 1)', '(v1, 0)', 0),
                                              ('(v5, 1)', '(v9, 0)', 0),
                                              ('(v5, 1)', '(v4, 0)', 0),
                                              ('(v5, 1)', '(v6, 0)', 0),
                                              ('(v5, 1)', '(v8, 0)', 0),  # add edges after human has intervened once
                                              ('(v1, 0)', '(v2, 0)', const_a),
                                              ('(v2, 0)', '(v9, 0)', 0),
                                              ('(v1, 0)', '(v3, 0)', 1),
                                              ('(v3, 0)', '(v6, 0)', 0),
                                              ('(v6, 0)', '(v5, 0)', 1),
                                              ('(v6, 0)', '(v7, 0)', 1),
                                              ('(v7, 0)', '(v8, 0)', 0),
                                              ('(v5, 0)', '(v4, 0)', 0),
                                              ('(v4, 0)', '(v2, 0)', 1)])

        # lets add a transition to accepting state form v4 and a transition to trap state from v5
        nstate_graph.add_edge('(v9, 0)', 'accept_all', weight=0)
        nstate_graph.add_edge('(v9, 1)', 'accept_all', weight=0)
        nstate_graph.add_edge('accept_all', 'accept_all', weight=0)
        nstate_graph.add_state_attribute('accept_all', 'player', 'eve')
        nstate_graph.add_accepting_state('accept_all')

        # adding trap state
        # nstate_graph.add_edge('trap', 'trap', weight=-1*math.inf)
        # nstate_graph.add_edge('trap', 'trap', weight=-3)
        # nstate_graph.add_state_attribute('trap', 'player', 'adam')

        # nstate_graph.add_edge('v5', 'trap', weight=0)
        nstate_graph.add_edge('(v8, 1)', 'trap', weight=0)
        nstate_graph.add_edge('(v8, 0)', 'trap', weight=0)
        nstate_graph.add_edge('trap', 'trap', weight=0)
        nstate_graph.add_state_attribute('trap', 'player', 'eve')

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

        nstate_graph.add_state_attribute('(v1, 1)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v2, 1)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v3, 1)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v4, 1)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v5, 1)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v6, 1)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v7, 1)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v8, 1)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v9, 1)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v1, 0)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v2, 0)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v3, 0)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v4, 0)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v5, 0)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v6, 0)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v7, 0)', 'player', 'adam')
        nstate_graph.add_state_attribute('(v8, 0)', 'player', 'eve')
        nstate_graph.add_state_attribute('(v9, 0)', 'player', 'eve')

        nstate_graph.add_initial_state('(v1, 1)')
        #
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
            self._instance = TwoPlayerGraph.build_running_ex(graph_name, config_yaml, save_flag)

        if plot:
            self._instance.plot_graph()

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:
        config_data = self.load_YAML_config_data(config_file_name)

        return config_data