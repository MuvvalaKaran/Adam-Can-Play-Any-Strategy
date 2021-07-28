import math
import copy
import multiprocessing
import warnings
import random
import sys
import operator

from joblib import Parallel, delayed
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from ..graph import Graph, graph_factory
from ..graph import TwoPlayerGraph
from ..graph import ProductAutomaton
from ..mpg_tool import MpgToolBox

# needed for multi-threading w' computation
NUM_CORES = multiprocessing.cpu_count()


class InfiniteRegretMinimizationStrategySynthesis:
    """
    This class implements Algo 1, 2, 3, and 4 as sepcified the pseudocode pdf file

    :param      graph:          A concrete instance of class TwoPlayerGraph or ProductAutomation
                                (depending on how you construct the Graph) on which we will be performing the
                                regret minimizing strategy synthesis
    :param      payoff          A concrete instance of the payoff function to be used. Currently the code has
                                inf, sup, liminf, limsup, mean and Cumulative payoff implementation.
    """
    def __init__(self,
                 graph: TwoPlayerGraph,
                 payoff) -> 'InfiniteRegretMinimizationStrategySynthesis':
        self.graph = graph
        self.payoff = payoff
        self.b_val: Optional[set] = None

    @property
    def b_val(self):
        return self._b_val

    @b_val.setter
    def b_val(self, value: set):
        self._b_val = value

    def infinite_reg_solver(self,
                            plot: bool = False,
                            plot_only_eve: bool = False,
                            go_fast: bool = True,
                            finite: bool = False):
        """
        A method to compute payoff when using a type of infinite payoff. The weights associated with the game are costs
        and are non-negative.

        :param plot:
        :param plot_only_eve:
        :return:
        """

        # compute w_prime
        w_prime = self.compute_W_prime(go_fast=go_fast, debug=False)

        g_hat = self.construct_g_hat(w_prime, game=None, finite=finite, debug=True,
                                     plot=False)

        mpg_g_hat_handle = MpgToolBox(g_hat, "g_hat")

        reg_dict, reg_val = mpg_g_hat_handle.compute_reg_val(go_fast=True, debug=False)
        # g_hat.plot_graph()
        self.plot_str_from_mgp(g_hat, reg_dict, only_eve=plot_only_eve, plot=plot)

    def _compute_cval_from_mpg(self, go_fast: bool, debug: bool):
        mpg_cval_handle = MpgToolBox(self.graph, "org_graph")
        return mpg_cval_handle.compute_cval(go_fast=go_fast, debug=debug)

    def compute_W_prime(self, org_graph: Optional[TwoPlayerGraph] = None, go_fast: bool = False, debug: bool = False):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
         alternate strategy game.
        """

        print("*****************Constructing W_prime*****************")

        if isinstance(org_graph, type(None)):
            org_graph = self.graph
        else:
            org_graph = org_graph

        coop_dict = self._compute_cval_from_mpg(go_fast=go_fast, debug=debug)

        w_prime: Dict[Tuple: float] = {}

        for edge in org_graph._graph.edges():

            # if the node belongs to adam, then the corresponding edge is assigned -inf
            if org_graph._graph.nodes(data='player')[edge[0]] == 'adam':
                w_prime.update({edge: -1 * math.inf})

            else:
                # a list to save all the alternate strategy cVals from a node and then selecting
                # the max of it
                tmp_cvals = []
                out_going_edge = set(org_graph._graph.out_edges(edge[0])) - set([edge])
                for alt_e in out_going_edge:
                    tmp_cvals.append(coop_dict[alt_e[1]])

                if len(tmp_cvals) != 0:
                    w_prime.update({edge: max(tmp_cvals)})
                else:
                    w_prime.update({edge: -1 * math.inf})

        self.b_val = set(w_prime.values())
        print(f"the values of b are {set(w_prime.values())}")

        return w_prime

    def _compute_cval(self, multi_thread: bool = False) -> Dict:
        """
        A method that pre computes all the cVals for every node in the graph and stores them in a dictionary.
        :return: A dictionary of cVal stores in dict
        """
        max_coop_val = defaultdict(lambda: -1)

        if not multi_thread:
            for n in self.graph._graph.nodes():
                max_coop_val[n] = self._compute_max_cval_from_v(n)

            return max_coop_val
        else:
            print("*****************Start Parallel Processing*****************")
            runner = Parallel(n_jobs=NUM_CORES, verbose=50)
            job = delayed(self._compute_max_cval_from_v)
            results = runner(job(n) for n in self.graph._graph.nodes())
            print("*****************Stop Parallel Processing*****************")

        for _n, _r in zip(self.graph._graph.nodes(), results):
            max_coop_val[_n] = _r

        return max_coop_val

    def _compute_max_cval_from_v(self, node: Tuple) -> float:
        """
        A helper method to compute the cVal from a given vertex (@node) for a give graph @graph
        :param graph: The graph on which would like to compute the cVal
        :param payoff_handle: instance of the @compute_value() to compute the cVal
        :param node: The node from which we would like to compute the cVal
        :return: returns a single max value. If multiple plays have the max_value then the very first occurance is returned
        """
        # construct a new graph with node as the initial vertex and compute loop_vals again
        # 1. remove the current init node of the graph
        # 2. add @node as the new init vertex
        # 3. compute the loop-vals for this new graph
        # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
        self.payoff.remove_attribute(self.payoff.get_init_node(), 'init')
        self.payoff.set_init_node(node)
        self.payoff.cycle_main()

        return self.payoff.compute_cVal(node)

    def _construct_g_b(self,
                       g_hat: TwoPlayerGraph,
                       org_game: TwoPlayerGraph,
                       b: float,
                       w_prime: Dict,
                       init_node: List[Tuple],
                       accp_node: List[Tuple]) -> None:
        """

        :param g_hat:
        :param b:
        :param w_prime:
        :param init_node:
        :param accp_node:
        :return:
        """

        # each node is dict with the node name as key and 'b' as its value
        g_hat.add_states_from([((n), b) for n in self.graph._graph.nodes()])

        assert (len(init_node) == 1), f"Detected multiple init nodes in the org graph: {[n for n in init_node]}. " \
                                      f"This should not be the case"

        # assign each node a player if it hasn't been initialized yet
        for n in g_hat._graph.nodes():

            if org_game._graph.has_node(n[0]):
                # add aps to each node in g_hat
                g_hat._graph.nodes[n]['ap'] = org_game._graph.nodes[n[0]].get('ap')

            # assign the nodes of G_b with v1 in it at n[0] to have a 'init' attribute
            if len(n) == 2 and n[0] == init_node[0][0]:
                g_hat._graph.nodes[n]['init'] = True

            # assign the nodes of G_b with 'accepting' attribute
            for _accp_n in accp_node:
                if len(n) == 2 and n[0] == _accp_n:
                    g_hat._graph.nodes[n]['accepting'] = True

            if g_hat._graph.nodes(data='player')[n] is None:
                if org_game._graph.nodes(data='player')[n[0]] == "adam":
                    g_hat._graph.nodes[n]['player'] = "adam"
                else:
                    g_hat._graph.nodes[n]['player'] = "eve"

        # a sample edge og g_hat: ((".","."),"."),((".","."),".") and
        # a sample edge of org_graph: (".", ""),(".", ".")
        for e in org_game._graph.edges():
            if w_prime[e] <= b:
                g_hat.add_edge(((e[0]), b), ((e[1]), b))

    def _construct_g_hat_nodes(self, g_hat: ProductAutomaton) -> ProductAutomaton:
        """
        A helper function that adds the nodes v0, v1 and vT that are part of g_hat graph
        :return: A updated instance of g_hat
        """

        g_hat.add_states_from(['v0', 'v1', 'vT'])

        g_hat.add_state_attribute('v0', 'player', 'adam')
        g_hat.add_state_attribute('v1', 'player', 'eve')
        g_hat.add_state_attribute('vT', 'player', 'eve')

        # add v0 as the initial node
        g_hat.add_initial_state('v0')

        # add the edges with the weights
        g_hat.add_weighted_edges_from([('v0', 'v0', 0),
                                       ('v0', 'v1', 0),
                                       ('vT', 'vT', -2 * abs(self.graph.get_max_weight()) - 1)])
        return g_hat

    def construct_g_hat(self,
                        w_prime: Dict[Tuple, float],
                        acc_min_edge_weight: bool = False,
                        acc_max_edge_weight: bool = False,
                        game: Optional[TwoPlayerGraph] = None,
                        finite: bool = False, debug: bool = False, plot: bool = False) -> TwoPlayerGraph:
        print("*****************Constructing G_hat*****************")
        # construct new graph according to the pseudocode 3

        G_hat: ProductAutomaton = graph_factory.get("ProductGraph",
                                                    graph_name="G_hat",
                                                    config_yaml="/config/G_hat",
                                                    save_flag=True)
        G_hat.construct_graph()

        # build g_hat
        G_hat = self._construct_g_hat_nodes(G_hat)

        # choose which game to construct g_hat from - the org game G to the shifted G_delta
        if isinstance(game, type(None)):
            org_game = self.graph
        else:
            org_game = game

        # add accepting states to g_hat
        accp_nodes = org_game.get_accepting_states()

        # compute the range of w_prime function
        w_set = set(w_prime.values()) - {-1 * math.inf}
        org_init_nodes = org_game.get_initial_states()

        # construct g_b
        for b in w_set - {math.inf}:
            self._construct_g_b(G_hat, org_game, b, w_prime, org_init_nodes, accp_nodes)

        # add edges between v1 of G_hat and init nodes(v1_b/ ((v1, 1), b) of graph G_b with edge weights 0
        # get init node of the org graph
        init_node_list: List[Tuple] = G_hat.get_initial_states()

        # add edge with weigh 0 from v1 to (v1,b)
        for _init_n in init_node_list:
            if isinstance(_init_n[0], tuple):
                _b_val: int = _init_n[0][-1]
                G_hat.add_weighted_edges_from([('v1', _init_n[0], -1*_b_val)])
                G_hat.remove_state_attr(_init_n[0], "init")

        # add edges with their respective weights; a sample edge ((".","."),"."),((".","."),".") for with gmin/gmax and
        # with ((".","."),(".", "."))
        for e in G_hat._graph.edges():
            # only add weights if hasn't been initialized
            if G_hat._graph[e[0]][e[1]][0].get('weight') is None:

                # an edge can only exist within a graph g_b
                assert (e[0][1] == e[1][1]), \
                    "Make sure that there only exist edge between nodes that belong to the same g_b"

                if acc_min_edge_weight and G_hat._graph.nodes[e[0]].get('accepting') is not None:
                    G_hat._graph[e[0]][e[1]][0]['weight'] = 0

                else:
                    G_hat._graph[e[0]][e[1]][0]['weight'] = self._w_hat_b(org_game=org_game,
                                                                          org_edge=(e[0][0], e[1][0]),
                                                                          b_value=e[0][1],
                                                                          finite=False)

        # for nodes that don't have any outgoing edges add a transition to the terminal node i.e 'T' in our case
        for node in G_hat._graph.nodes():
            if G_hat._graph.out_degree(node) == 0:

                if acc_max_edge_weight:
                    # if the node belongs to the accepting state then add a self-loop to itself

                    if G_hat._graph.nodes[node].get('accepting') is not None:
                        G_hat.add_weighted_edges_from([(node, node, 0)])
                        continue

                # add transition to the terminal node
                G_hat.add_weighted_edges_from([(node, 'vT', 0)])

        if debug:
            print(f"# of G_hat nodes: {len(list(G_hat._graph.nodes()))}")
            print(f"# of G_hat edges: {len(list(G_hat._graph.edges()))}")

        if plot:
            G_hat.plot_graph()

        # after constructing g_hat, we need to ensure that all the absorbing states in g_hat have sel-loop transitions
        # and transitions from the sys nodes with edge weight 0
        _accp_states = G_hat.get_accepting_states()
        _trap_states = G_hat.get_trap_states()

        # the weights of vT is different for finite and infinite case
        # if not finite:
        #     for _s in _accp_states + _trap_states:
        #         if _s != "vT":
        #             for _pre_s in G_hat._graph.predecessors(_s):
        #                 # avoid trap state self loops
        #                 if _pre_s not in _trap_states:
        #                     G_hat._graph[_pre_s][_s][0]['weight'] = 0

        return G_hat

    def _extract_g_b_graph(self, b_val: int, g_hat: TwoPlayerGraph) -> 'TwoPlayerGraph()':
        """
        A method that extracts the g_b graph from g_hat graph given the b_val
        :param b_val:
        :return:
        """

        if not (isinstance(b_val, int) and isinstance(b_val, float)):
            warnings.warn("Please make sure b_val is an int or float value")

        if b_val not in self.b_val:
            warnings.warn(f"Make sure you enter a valid b value."
                          f"The set of valid b value is {[_b for _b in self.b_val]}")
            sys.exit(-1)

        _g_b_init_node = None
        _g_b_accp_node = None
        # we use this special loop as the the g_b init node is removed as the init node during g_hat construction
        for _n in g_hat._graph.successors("v1"):
            if _n[1] == b_val:
                _g_b_init_node = _n
                break

        for _n in g_hat.get_accepting_states():
            if _n[1] == b_val:
                _g_b_accp_node = _n
                break

        _g_b_nodes = set()
        for _n in g_hat._graph.nodes():
            if isinstance(_n, tuple) and _n[1] == b_val:
                _g_b_nodes.add(_n)

        _g_b_graph = graph_factory.get("TwoPlayerGraph",
                                       graph_name="g_b_graph",
                                       config_yaml="/config/g_b_graph",
                                       save_flag=True,
                                       pre_built=False,
                                       from_file=False,
                                       plot=False)

        for _n in _g_b_nodes:
            _player = g_hat.get_state_w_attribute(_n, "player")
            _g_b_graph.add_state(_n, player=_player)

        for _n in _g_b_graph._graph.nodes():
            for _next_n in g_hat._graph.successors(_n):
                if _next_n in _g_b_nodes:
                    _weight = g_hat.get_edge_weight(_n, _next_n)
                    _g_b_graph.add_edge(_n, _next_n, weight=_weight)

        # add initial state and accepting state attribute
        if _g_b_init_node is None or _g_b_accp_node is None:
            warnings.warn(f"Could not find the initial state or the accepting state of graph g_{b_val}")

        _g_b_graph.add_accepting_state(_g_b_accp_node)
        _g_b_graph.add_initial_state(_g_b_init_node)

        _g_b_graph.add_state("vT", player="eve")

        # for states that do not have any outgoing edges, add a transition to the terminal state with edge weight 0
        for _n in _g_b_graph._graph.nodes():
            if len(list(_g_b_graph._graph.successors(_n))) == 0:
                _g_b_graph.add_edge(_n, "vT", weight=0)

        return _g_b_graph

    def _w_hat_b(self,
                 org_game: TwoPlayerGraph,
                 org_edge: Tuple[Tuple[str, str], Tuple[str, str]],
                 b_value: float,
                 finite: bool) -> float:
        """
        an helper function that returns the w_hat value for a g_b graph : w_hat(e) = w(e) - b
        :param org_edge: edges of the format ("v1", "v2") or Tuple of tuples
        :param b_value:
        :return:
        """
        try:
            if finite:
                _sub_val = b_value/(len(org_game._graph.nodes()) - 1)
                return org_game._graph[org_edge[0]][org_edge[1]][0].get('weight') - _sub_val
            else:
                # if b_value == -3:
                #     return org_game._graph[org_edge[0]][org_edge[1]][0].get('weight') - b_value - 4
                # else:
                # return org_game._graph[org_edge[0]][org_edge[1]][0].get('weight') - b_value
                return org_game._graph[org_edge[0]][org_edge[1]][0].get('weight')
        except KeyError:
            print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                  "original graph has been modified and the edge {} does not exist".format(org_edge))

    def _add_strategy_flag(self,
                           g_hat: Union[TwoPlayerGraph, ProductAutomaton, Graph],
                           combined_strategy):
        """
        A helper method that adds a strategy attribute to the nodes of g_hat that belong to the strategy dict computed.

        Effect : Adds a new attribute "strategy" as False and loops over the dict and updated attribute
         to True if that node exists in the strategy dict.
        :param g_hat: The graph on which we compute the regret minimizing strategy
        :param combined_strategy: Combined dictionary of sys(eve)'s and env(adam)'s strategy
        :return:
        """

        for curr_node, next_node in combined_strategy.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    g_hat._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                g_hat._graph.edges[curr_node, next_node, 0]['strategy'] = True

    def _add_strategy_flag_only_eve(self,
                                    g_hat: Union[TwoPlayerGraph, ProductAutomaton, Graph],
                                    combined_strategy):
        """
        A helper method that adds a strategy attribute to the nodes of g_hat that belong to the strategy dict computed.

        Effect : Adds a new attribute "strategy" as False and loops over the dict and updated attribute
         to True if that node exists in the strategy dict and belongs to eve ONLY.

        :param g_hat: The graph on which we compute the regret minimizing strategy
        :param combined_strategy: Combined dictionary of sys(eve)'s and env(adam)'s strategy
        :return:
        """

        for curr_node, next_node in combined_strategy.items():
            if g_hat._graph.nodes[curr_node].get("player") == "eve":
                if isinstance(next_node, list):
                    for n_node in next_node:
                        g_hat._graph.edges[curr_node, n_node, 0]['strategy'] = True
                else:
                    g_hat._graph.edges[curr_node, next_node, 0]['strategy'] = True

    def plot_str_from_mgp(self,
                          g_hat: TwoPlayerGraph,
                          str_dict: Dict,
                          only_eve: bool = False,
                          plot: bool = False) -> Dict:
        """
        A helper method that plots all the VALID strategies computed on g_hat on g_hat. It then maps back the
         least regret strategy back to the original strategy.
        :return:
        """

        g_hat.set_edge_attribute('strategy', False)

        if only_eve:
            self._add_strategy_flag_only_eve(g_hat, str_dict)

        else:
            self._add_strategy_flag(g_hat, str_dict)

        # Map back strategies from g_hat to the original abstraction
        org_str = self._from_str_mpg_to_str(str_dict)

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, org_str)

        else:
            self._add_strategy_flag(self.graph, org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

        return org_str