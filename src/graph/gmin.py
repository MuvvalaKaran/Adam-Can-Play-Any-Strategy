from .two_player_graph import TwoPlayerGraph
from .graph import Graph
from .trans_sys import FiniteTransSys
from .dfa import DFAGraph
from src.factory.builder import Builder


class GMinGraph(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._trans_sys = None
        self._auto_graph = None
        # self._graph_name = graph_name
        # self._config_yaml = config_yaml
        # self._save_flag = save_flag
        super.__init__(graph_name, config_yaml, save_flag)

    def construct_graph(self):
        super().construct_graph()

    @classmethod
    def construct_gmin_from_grapg(cls, graph, debug: bool = False):

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
                graph.add_state((n[0][0], n[1]))
                graph.add_state_attribute((n[0][0], n[1]), 'player', 'eve')

            else:
                graph.add_state((n[0][0], n[1]))
                graph.add_state_attribute((n[0][0], n[1]), 'player', 'adam')

            # if the node has init attribute and n[1] == W then add it to the init vertex in Gmin
            if n[0][1].get('init') and n[1] == W:
                # Gmin.nodes[(n[0][0], n[1])]['init'] = True
                graph.add_initial_state((n[0][0], n[1]))
            if n[0][1].get('accepting'):
                graph.add_accepting_state((n[0][0], n[1]))

        if debug:
            graph.print_nodes()

        # constructing edges as per the requirement mentioned in the doc_string
        for parent in graph._graph.nodes:
            for child in graph._graph.nodes:
                if graph._graph.has_edge(parent[0], child[0]):
                    if float(child[1]) == min(float(parent[1]),
                                              float(graph._graph.get_edge_data(parent[0],
                                                                                         child[0])[0]['weight'])):
                        graph._graph.add_edge(parent, child, weight=child[1])

        if debug:
            graph.print_edges()

        # if plot:
        #     graph.plot_graph()

        return graph


class GMinBuilder(Builder):

    def __init__(self):

        # call the parent class constructor
        Builder.__init__(self)

        self.trans_sys: FiniteTransSys = None
        self.automaton: DFAGraph = None

    def __call__(self,
                 graph: Graph,
                 graph_name,
                 config_yaml,
                 manual_constr: bool = False,
                 save_flag:bool = False) -> 'GminGraph':
        """
        A method that returns an initialized Gmin instance given two player graph
        :param graph: A two player graph which could be of type TwoPlayerGraph or ProductGraph depending on
        how it was created
        :param graph_name:
        :param config_yaml:
        :param manual_constr:   True:  if we want to use the manually constructed two player graph
                                False: construct a product graph given a transition system and LTL formula
        :param save_flag:
        :return:
        """

        gmin_graph = GMinBuilder()