from .two_player_graph import TwoPlayerGraph
from .graph import Graph
from .product import ProductAutomaton, ProductBuilder
from .trans_sys import FiniteTransSys, TransitionSystemBuilder
from .dfa import DFAGraph, DFABuilder
from src.factory.builder import Builder


class GMinGraph(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._trans_sys = None
        self._auto_graph = None
        # self._graph_name = graph_name
        # self._config_yaml = config_yaml
        # self._save_flag = save_flag
        TwoPlayerGraph.__init__(self, graph_name, config_yaml, save_flag)

    def construct_graph(self):
        super().construct_graph()

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
        max_edge = max(dict(graph._graph.edges).items(), key=lambda x: x[1]['weight'])
        W: str = max_edge[1].get('weight')

        # assign nodes to Gmin with player as attributes to each node
        for n in V_prime:
            if n[0][1]['player'] == "eve":
                gmin_graph.add_state((n[0][0], n[1]))
                gmin_graph.add_state_attribute((n[0][0], n[1]), 'player', 'eve')

            else:
                gmin_graph.add_state((n[0][0], n[1]))
                gmin_graph.add_state_attribute((n[0][0], n[1]), 'player', 'adam')

            # if the node has init attribute and n[1] == W then add it to the init vertex in Gmin
            if n[0][1].get('init') and n[1] == W:
                # Gmin.nodes[(n[0][0], n[1])]['init'] = True
                gmin_graph.add_initial_state((n[0][0], n[1]))
            if n[0][1].get('accepting'):
                gmin_graph.add_accepting_state((n[0][0], n[1]))

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in gmin_graph._graph.nodes:
            for child in gmin_graph._graph.nodes:
                if graph._graph.has_edge(parent[0], child[0]):
                    if float(child[1]) == min(float(parent[1]),
                                              float(graph._graph.get_edge_data(parent[0],
                                                                                         child[0])[0]['weight'])):
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

        self.trans_sys: FiniteTransSys = None
        self.automaton: DFAGraph = None
        self.scLTL: str = None

    def __call__(self,
                 graph: Graph,
                 graph_name: str,
                 config_yaml: str,
                 trans_sys: FiniteTransSys,
                 dfa: DFAGraph,
                 # sc_ltl: str = "",
                 manual_constr: bool = False,
                 debug: bool = False,
                 save_flag: bool = False,
                 # use_alias: bool = False,
                 plot: bool = False,
                 # human_intervention: int = 1,
                 # absorbing: bool = False,
                 pre_built: bool = True) -> 'GMinGraph':
        """
        A method that returns an initialized Gmin instance given two player graph
        :param graph:           A two player graph which could be of type TwoPlayerGraph or ProductGraph depending on
                                how it was created
        :param graph_name:
        :param config_yaml:
        :param trans_sys:
        :param manual_constr:   True:  if we want to use the manually constructed two player graph
                                False: construct a product graph given a transition system and LTL formula
        :param debug:
        :param save_flag:
        :param plot:
        # :param human_intervention:
        :param pre_built:
        # :param plot_raw_ts:
        :return:
        """

        self.gmin_graph = GMinGraph(graph_name, config_yaml, save_flag)

        if manual_constr:
            if not (isinstance(graph, ProductAutomaton) or isinstance(graph, TwoPlayerGraph)):
                raise TypeError(
                    f"Graph should either be of type {ProductAutomaton.__name__} or {TwoPlayerGraph.__name__}")

            self._instance = self._from_prebuilt_graph(graph,
                                                       graph_name=graph_name,
                                                       config_yaml=config_yaml,
                                                       save_flag=save_flag,
                                                       debug=debug,
                                                       plot=plot)
        else:
            # if not (isinstance(sc_ltl, str) or sc_ltl == ""):
            #     raise TypeError(f"Please ensure that the ltl formula is of type string is not empty.")

            if not isinstance(trans_sys, FiniteTransSys):
                raise TypeError(f"Please ensure that the transition system is of type {FiniteTransSys.__name__}")

            self._from_ts_and_dfa(trans_sys=trans_sys,
                                  dfa=dfa,
                                  save_flag=save_flag,
                                  plot=plot,
                                  debug=debug)

        return self._instance

    def _from_prebuilt_graph(self, graph: Graph,
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

    def _from_ts_and_dfa(self,
                         trans_sys: FiniteTransSys,
                         dfa: DFAGraph,
                         save_flag: bool,
                         plot: bool,
                         debug: bool):
        """
        A method to return an concrete instance of GMin given a dfa  and a Transition system - the abstraction
        :param raw_trans_sys:
        :param sc_ltl:
        :param use_alias:
        :param plot:
        :param human_intervention:
        :param absorbing:
        :return:
        """

        # call the transition system builder method
        # trans_sys_builder = TransitionSystemBuilder()
        # trans_sys = trans_sys_builder(raw_trans_sys=raw_trans_sys,
        #                               graph_name='trans_sys',
        #                               config_yaml='config/trans_sys',
        #                               save_flag=save_flag,
        #                               pre_built=False,
        #                               debug=debug,
        #                               plot=plot,
        #                               human_intervention=human_intervention,
        #                               plot_raw_ts=plot_raw_ts)

        # build the automaton based on the LTL formula
        # dfa_builder = DFABuilder()
        # dfa = dfa_builder()

        # build the product automaton
        product_builder = ProductBuilder()
        product = product_builder()

        return self.gmin_graph.construct_gmin_from_graph(product,
                                                         graph_name="gmin_ts",
                                                         config_yaml="config/gmin_ts",
                                                         save_flag=save_flag,
                                                         debug=debug)
