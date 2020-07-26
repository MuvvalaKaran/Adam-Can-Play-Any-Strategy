import networkx as nx
import math

# local packages
from .base import Graph
from src.factory.builder import Builder

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

    def get_max_weight(self) -> str:
        max_weight: int = 0
        # loop through all the edges and return the max weight
        for _e in self._graph.edges.data("weight"):
            if abs(float(_e[2])) != math.inf and abs(float(_e[2])) > abs(max_weight):
                max_weight = float(_e[2])

        return str(max_weight)

    @classmethod
    def build_running_ex(cls: 'TwoPlayerGraph',
                         graph_name: str,
                         config_yaml: str,
                         save_flag: bool = False) \
            -> 'TwoPlayerGraph()':
        """
        A class method that constructs the sample three state graph for you
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :return: An concrete instance of the built three state grpah as described in the configuration above
        """

        nstate_graph = TwoPlayerGraph(graph_name=graph_name, config_yaml=config_yaml, save_flag=save_flag)
        nstate_graph.construct_graph()

        nstate_graph.add_weighted_edges_from([('v1', 'v2', '1'),
                                                  ('v2', 'v1', '-1'),
                                                  ('v1', 'v3', '1'),
                                                  ('v3', 'v3', '0.5'),
                                                  ('v3', 'v5', '1'),
                                                  ('v2', 'v4', '2'),
                                                  ('v4', 'v4', '2'),
                                                  ('v5', 'v5', '1')])

        nstate_graph.add_state_attribute('v1', 'player', 'eve')
        nstate_graph.add_state_attribute('v2', 'player', 'adam')
        nstate_graph.add_state_attribute('v3', 'player', 'adam')
        nstate_graph.add_state_attribute('v4', 'player', 'eve')
        nstate_graph.add_state_attribute('v5', 'player', 'eve')

        nstate_graph.add_initial_state('v1')

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
                 pre_built: bool = False,
                 plot: bool = False) -> TwoPlayerGraph:
        """
        Return an initialized TwoPlayerGraph instance given the configuration data
        :param graph_name : Name of the graph
        :return: A concrete/active instance of the TwoPlayerGraph
        """
        self._instance = TwoPlayerGraph(graph_name, config_yaml, save_flag)
        self._instance.construct_graph()

        if pre_built:
            self._instance = TwoPlayerGraph.build_running_ex(graph_name, config_yaml, save_flag)

        if plot:
            self._instance.plot_graph()

        return self._instance