import warnings

from .two_player_graph import TwoPlayerGraph
from .base import Graph
from .product import ProductAutomaton, ProductBuilder
from .trans_sys import FiniteTransSys, TransitionSystemBuilder
from .dfa import DFAGraph, DFABuilder
from ..factory.builder import Builder


class GMinGraph(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._trans_sys = None
        self._auto_graph = None
        TwoPlayerGraph.__init__(self, graph_name, config_yaml, save_flag)

    @classmethod
    def construct_gmin_from_graph(cls, graph: Graph,
                                  graph_name: str,
                                  config_yaml: str,
                                  save_flag: bool = False,
                                  debug: bool = False,
                                  plot: bool = False):

        gmin_graph = GMinGraph(graph_name, config_yaml, save_flag)
        gmin_graph.construct_graph()

        # construct new set of states V'
        V_prime = [(v, str(w)) for v in graph._graph.nodes.data() for _, _, w in
                   graph._graph.edges.data('weight')]

        # find the maximum weight in the og graph(G)
        # specifically adding self.graph.edges.data('weight') to a create to tuple where the
        # third element is the weight value
        _edge_w_max_weight = max(dict(graph._graph.edges).items(), key=lambda x: x[1]['weight'])
        _max_weight: str = _edge_w_max_weight[1].get('weight')

        # assign nodes to Gmin with player as attributes to each node
        for n in V_prime:
            if n[0][1]['player'] == "eve":
                gmin_graph.add_state((n[0][0], n[1]))
                gmin_graph.add_state_attribute((n[0][0], n[1]), 'player', 'eve')

            else:
                gmin_graph.add_state((n[0][0], n[1]))
                gmin_graph.add_state_attribute((n[0][0], n[1]), 'player', 'adam')

            # if the node has init attribute and n[1] == W then add it to the init vertex in Gmin
            if n[0][1].get('init') and n[1] == _max_weight:
                gmin_graph.add_initial_state((n[0][0], n[1]))
            if n[0][1].get('accepting'):
                gmin_graph.add_accepting_state((n[0][0], n[1]))

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in gmin_graph._graph.nodes:
            for child in gmin_graph._graph.nodes:
                if graph._graph.has_edge(parent[0], child[0]):
                    if child[1] == min(parent[1],
                                       graph._graph.get_edge_data(parent[0], child[0])[0]['weight']):
                        gmin_graph._graph.add_edge(parent, child, weight=child[1])

        if debug:
            gmin_graph.print_nodes()
            gmin_graph.print_edges()

        if plot:
            gmin_graph.plot_graph()

        return gmin_graph


class GMinBuilder(Builder):

    def __init__(self):

        # call the parent class constructor
        Builder.__init__(self)

        # self.trans_sys: FiniteTransSys = None
        # self.automaton: DFAGraph = None
        # self.scLTL: str = None

    def __call__(self,
                 graph: Graph,
                 graph_name: str,
                 config_yaml: str,
                 debug: bool = False,
                 save_flag: bool = False,
                 plot: bool = False) -> 'GMinGraph':
        """
        A method that returns an initialized GMinGraph instance given a two player graph or a FiniteTransition System
        :param graph:           A two player graph which could be of type TwoPlayerGraph or ProductGraph depending on
                                how it was created
        :param graph_name:
        :param config_yaml:
        :param debug:
        :param save_flag:
        :param plot:
        :return:
        """

        self.gmin_graph = GMinGraph(graph_name, config_yaml, save_flag)

        if not (isinstance(graph, FiniteTransSys) or isinstance(graph, TwoPlayerGraph)):
            raise TypeError(
                f"Graph should either be of type {ProductAutomaton.__name__} or {TwoPlayerGraph.__name__}")

        # if pass in a product automaton, which is constructed using absorbing flag then,
        # we cannot construct GMin or GMax, as absorbing states do not belong to any state

        if isinstance(graph, ProductAutomaton):
            warnings.warn(f"Passed a Product Automaton. GMin construction will fail if it was constructed" \
                          f" using absorbing flag as true")

        self._instance = self._from_ts(graph,
                                       graph_name=graph_name,
                                       config_yaml=config_yaml,
                                       save_flag=save_flag,
                                       debug=debug,
                                       plot=plot)

        return self._instance

    def _from_ts(self,
                 graph: Graph,
                 graph_name: str,
                 config_yaml: str,
                 save_flag: bool,
                 debug: bool,
                 plot: bool):
        """
        A method that return an concrete instance of GMin from a pre built two player graph
        :param graph: A two player graph which is to be converted to a gmin according to theory in the paper
        :param debug: A flag to print all information regarding the graph while it is constructed
        :return: An active instance of the gmin graph
        """

        return self.gmin_graph.construct_gmin_from_graph(graph,
                                                         graph_name,
                                                         config_yaml,
                                                         save_flag,
                                                         debug,
                                                         plot)