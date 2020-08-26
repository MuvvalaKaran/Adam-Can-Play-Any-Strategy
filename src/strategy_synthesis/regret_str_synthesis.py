import math
import copy
import multiprocessing
import warnings
import random
import sys
import operator

import numpy as np
from numpy import ndarray
from joblib import Parallel, delayed
from _collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from src.graph import Graph, graph_factory
from src.graph import TwoPlayerGraph
from src.graph import ProductAutomaton
from src.payoff import Payoff
from src.mpg_tool import MpgToolBox


# import local value iteration solver
from .value_iteration import ValueIteration

# needed for multi-threading w' computation
NUM_CORES = multiprocessing.cpu_count()


class RegretMinimizationStrategySynthesis:
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
                 payoff: Payoff) -> 'RegretMinimizationStrategySynthesis()':
        self.graph = graph
        self.payoff = payoff
        self.b_val: Optional[set] = None

    @property
    def b_val(self):
        return self._b_val

    @b_val.setter
    def b_val(self, value: set):
        self._b_val = value

    def compute_W_prime_finite(self,  multi_thread: bool = False):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
        alternate strategy game. This is slightly different in finite payoffs in that we add that edge (u,v')
        that we skip in infinite payoff computation. In that case its fine, as it's the edges that we encounter
        infinitely often. But, for finite payoff computation skipping edge will effect the W_prime. Computation

        Changes : We manually add the edge weight associated with (u, v') that we skip.
        """

        print("*****************Constructing W_prime finite*****************")

        mcr_solver = ValueIteration(self.graph, competitve=False)
        mcr_solver.solve(debug=False, plot=False)
        INT_MAX_VAL = 2147483647
        # as the edge weight in the value iteration are all positive, we need to manually add negative weigh to them
        val_dict = mcr_solver.state_value_dict

        # also the MAX value used in the ValueIteration algorithm is a finite value. So we will replace them with -inf
        for _s, _s_val in val_dict.items():
            _new_cost = _s_val
            if _s_val > 0:
                _new_cost = -1 * _s_val

                if _new_cost == -1 * INT_MAX_VAL:
                    _new_cost = -1 * math.inf

            val_dict[_s] = _new_cost

        w_prime: Dict[Tuple: float] = {}

        for edge in self.graph._graph.edges():
            _u = edge[0]
            # _v = edge[1]
            # if the node belongs to adam, then the corresponding edge is assigned -inf
            if self.graph._graph.nodes(data='player')[_u] == 'adam':
                w_prime.update({edge: -1 * math.inf})

            else:
                _state_cvals = []
                out_going_edges = set(self.graph._graph.out_edges(_u)) - {edge}

                for _alt_e in out_going_edges:
                    _v_prime = _alt_e[1]
                    _curr_w = self.graph.get_edge_weight(_u, _v_prime)
                    _state_cost = val_dict[_v_prime]
                    _total_weight = _curr_w + _state_cost
                    _state_cvals.append(_total_weight)

                if len(_state_cvals) != 0:
                    w_prime.update({edge: max(_state_cvals)})
                else:
                    w_prime.update({edge: -1 * math.inf})

        self.b_val = set(w_prime.values())
        print(f"the value of b are {set(w_prime.values())}")

        return w_prime

    def __check_edge_in_cval_play(self, edge: Tuple, play: List[Tuple]):
        """
        A helper method to check if an edge already exists in the play associated with a node in the coop dict
        :return: Return True id that exists else false
        """

        try:
            u_idx = play.index(edge[0])
            if play[u_idx + 1] == edge[1]:
                return True
        except ValueError:
            return False
        return False

    def _compute_cval_finite(self, multi_thread: bool = False) -> Dict:
        """
        A method that pre computes all the cVals for every node in the graph and stores them in a dictionary.
        :return: A dictionary of cVal stores in dict
        """
        max_coop_val = defaultdict(lambda: -1)
        if not multi_thread:
            for n in self.graph._graph.nodes():
                max_coop_val[n] = (self._compute_max_cval_from_v(n))

            return max_coop_val
        else:
            print("*****************Start Parallel Processing*****************")
            runner = Parallel(n_jobs=NUM_CORES, verbose=50)
            job = delayed(self._compute_max_cval_from_v_finite)
            results = runner(job(n) for n in self.graph._graph.nodes())
            print("*****************Stop Parallel Processing*****************")

        for _n, _r in zip(self.graph._graph.nodes(), results):
            max_coop_val[_n] = _r

        return max_coop_val

    def _compute_max_cval_from_v_finite(self, node: Tuple) -> float:
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

    def _compute_cval_from_mpg(self, go_fast: bool, debug: bool):
        mpg_cval_handle = MpgToolBox(self.graph, "org_graph")
        return mpg_cval_handle.compute_cval(go_fast=go_fast, debug=debug)

    def compute_W_prime(self, go_fast: bool = False, debug: bool = False):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
         alternate strategy game.
        """

        print("*****************Constructing W_prime*****************")
        coop_dict = self._compute_cval_from_mpg(go_fast=go_fast, debug=debug)

        w_prime: Dict[Tuple: float] = {}

        for edge in self.graph._graph.edges():

            # if the node belongs to adam, then the corresponding edge is assigned -inf
            if self.graph._graph.nodes(data='player')[edge[0]] == 'adam':
                w_prime.update({edge: -1 * math.inf})

            else:
                # a list to save all the alternate strategy cVals from a node and then selecting
                # the max of it
                tmp_cvals = []
                out_going_edge = set(self.graph._graph.out_edges(edge[0])) - set([edge])
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

    def _construct_g_b(self, g_hat: TwoPlayerGraph,
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

            if self.graph._graph.has_node(n[0]):
                # add aps to each node in g_hat
                g_hat._graph.nodes[n]['ap'] = self.graph._graph.nodes[n[0]].get('ap')

            # assign the nodes of G_b with v1 in it at n[0] to have a 'init' attribute
            if len(n) == 2 and n[0] == init_node[0][0]:
                g_hat._graph.nodes[n]['init'] = True

            # assign the nodes of G_b with 'accepting' attribute
            for _accp_n in accp_node:
                if len(n) == 2 and n[0] == _accp_n:
                    g_hat._graph.nodes[n]['accepting'] = True

            if g_hat._graph.nodes(data='player')[n] is None:
                if self.graph._graph.nodes(data='player')[n[0]] == "adam":
                    g_hat._graph.nodes[n]['player'] = "adam"
                else:
                    g_hat._graph.nodes[n]['player'] = "eve"

        # a sample edge og g_hat: ((".","."),"."),((".","."),".") and
        # a sample edge of org_graph: (".", ""),(".", ".")
        for e in self.graph._graph.edges():
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

    def _construct_g_hat_nodes_finite(self, g_hat: ProductAutomaton) -> ProductAutomaton:
        """
        A helper function that adds the nodes v0, v1 and vT that are part of g_hat graph when
        using finite payoff

        In this construction, the terminal node will have the edge weight of -inf
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
                                       ('vT', 'vT', -1 * math.inf)])
        return g_hat

    def construct_g_hat(self,
                        w_prime: Dict[Tuple, float],
                        acc_min_edge_weight: bool = False,
                        acc_max_edge_weight: bool = False,
                        finite: bool = False, debug: bool = False, plot: bool = False) -> TwoPlayerGraph:
        print("*****************Constructing G_hat*****************")
        # construct new graph according to the pseudocode 3

        G_hat: ProductAutomaton = graph_factory.get("ProductGraph",
                                                    graph_name="G_hat",
                                                    config_yaml="config/G_hat",
                                                    save_flag=True)
        G_hat.construct_graph()

        # build g_hat
        if finite:
            G_hat = self._construct_g_hat_nodes_finite(G_hat)
        else:
            G_hat = self._construct_g_hat_nodes(G_hat)

        # add accepting states to g_hat
        accp_nodes = self.graph.get_accepting_states()

        # compute the range of w_prime function
        w_set = set(w_prime.values()) - {-1 * math.inf}
        org_init_nodes = self.graph.get_initial_states()

        # construct g_b
        for b in w_set - {math.inf}:
            self._construct_g_b(G_hat, b, w_prime, org_init_nodes, accp_nodes)

        # add edges between v1 of G_hat and init nodes(v1_b/ ((v1, 1), b) of graph G_b with edge weights 0
        # get init node of the org graph
        init_node_list: List[Tuple] = G_hat.get_initial_states()

        # add edge with weigh 0 from v1 to (v1,b)
        for _init_n in init_node_list:
            if isinstance(_init_n[0], tuple):
                G_hat.add_weighted_edges_from([('v1', _init_n[0], 0)])
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
                    G_hat._graph[e[0]][e[1]][0]['weight'] = self._w_hat_b(org_edge=(e[0][0], e[1][0]),
                                                                          b_value=e[0][1])

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
                                       config_yaml="config/g_b_graph",
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

        return _g_b_graph

    def compute_cumulative_reg(self, g_hat: TwoPlayerGraph):
        mcr_solver = ValueIteration(g_hat, competitve=True)
        mcr_solver.solve(debug=True, plot=False)
        value_dict = mcr_solver.state_value_dict
        INT_MAX_VAL = 2147483647

        _g_b_init_nodes = [i for i in g_hat._graph.successors("v1")]
        # find out which _g_b_init_nodes have finite value and the pick the least one of them as the b_val

        _g_b_init_state_values = []
        for _n in _g_b_init_nodes:
            if value_dict[_n] != INT_MAX_VAL:
                _g_b_init_state_values.append((_n[1], value_dict[_n]))

        # if v1 has a value other than INT_MAX_VAL then we have a winning strategy that guarantees task completion
        # and the value will be the state cost associated with v1 or the next init state (the edge between v1 and g_b_init_node is 0).
        # if v1 is = MAX_INT_VAL then it means that we cannot guarantee winning. In this case, lets enter the g_b_max
        # game venue and compute a str according to the Min cost reachability theory

        # reg in this case can be defined as 0 if v1 is finite (i.e have a winning strategy) else, we roll out the game
        # in g_b_max and see where eventually end up. The val of that state will be our regret.
        if value_dict["v1"] != INT_MAX_VAL:
            print("There exists a winning strategy. The Reg is 0")
            # pick the g_b graph the str transits to - this may or may not be g_b_max
            if len(_g_b_init_state_values) != 0:
                _b_val = min(_g_b_init_state_values, key=operator.itemgetter(1))[0]
            else:
                warnings.warn("A winning strategy exists but none of the initial states in g_b(s) have finite value."
                              "This is a huge problem!")
                sys.exit(-1)

        else:
            print("There does not exists a winning strategy")
            _b_val = max(self.b_val)
        _g_b_graph = self._extract_g_b_graph(_b_val, g_hat)

        _solver = ValueIteration(_g_b_graph, competitve=True)
        _solver.solve()
        str_dict = _solver.compute_strategies(max_prefix_len=0)

        print("Debugging")

        return str_dict

    def _w_hat_b(self,
                 org_edge: Tuple[Tuple[str, str], Tuple[str, str]],
                 b_value: float) -> float:
        """
        an helper function that returns the w_hat value for a g_b graph : w_hat(e) = w(e) - b
        :param org_edge: edges of the format ("v1", "v2") or Tuple of tuples
        :param b_value:
        :return:
        """
        try:
            return self.graph._graph[org_edge[0]][org_edge[1]][0].get('weight') + b_value
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

    def _from_str_mpg_to_str(self, combined_str: Dict):
        original_str = {}
        # follow the strategy from the mpg toolbox
        node_stack = []
        curr_node = "v1"
        b_val = combined_str[curr_node][1]

        for u_node, v_node in combined_str.items():
            if self.graph._graph.has_node(u_node[0]):
                if u_node[1] == b_val and v_node[1] == b_val:
                    if self.graph._graph.has_edge(u_node[0], v_node[0]):
                            original_str.update({u_node[0]: v_node[0]})
        return original_str

    def get_controls_from_str(self, str_dict: Dict, debug: bool = False) -> List[str]:
        """
        A helper method to return a list of actions (edge labels) associated with the strategy found
        :param str_dict: The regret minimizing strategy
        :return: A sequence of labels that to be executed by the robot
        """

        start_state = self.graph.get_initial_states()[0][0]
        accepting_state = self.graph.get_accepting_states()[0]
        trap_state = self.graph.get_trap_states()[0]
        control_sequence = []

        curr_state = start_state
        next_state = str_dict[curr_state]
        # if self.graph._graph.nodes[next_state].get("player") == "adam":
        #     curr_state = str_dict[next_state]
        #     next_state = str_dict[curr_state]
        #     x, y = curr_state[0][0].split("(")[1].split(")")[0].split(",")
        #     control_sequence.append(("rand", np.array([int(x), int(y)])))
        # else:
        control_sequence.append(self.graph.get_edge_attributes(curr_state, next_state, 'actions'))
        while curr_state != next_state:
            if next_state == accepting_state or next_state == trap_state:
                break

            curr_state = next_state
            next_state = str_dict[curr_state]
            # if self.graph._graph.nodes[next_state].get("player") == "adam":
            #     curr_state = str_dict[next_state]
            #     next_state = str_dict[curr_state]
            #     x, y = curr_state[0][0].split("(")[1].split(")")[0].split(",")
            #     control_sequence.append(("rand", np.array([int(x), int(y)])))
            # else:
            control_sequence.append(self.graph.get_edge_attributes(curr_state, next_state, 'actions'))

        if debug:
            print([_n for _n in control_sequence])

        return control_sequence

    def _epsilon_greedy_choose_action(self,
                                      _human_state: tuple,
                                      str_dict: Dict[tuple, tuple],
                                      epsilon: float, _absoring_states: List[tuple],
                                      human_can_intervene: bool = False) -> Tuple[tuple, bool]:
        """
        Choose an action according to epsilon greedy algorithm

        Using this policy we either select a random human action with epsilon probability and the human can select the
        optimal action (as given in the str dict if any) with 1-epsilon probability.

        This method returns a human action based on the above algorithm.
        :param _human_state: The current state in the game from which we need to pick an action
        :return: Tuple(next_state, flag) . flag = True if human decides to take an action.
        """

        if self.graph.get_state_w_attribute(_human_state, "player") != "adam":
            warnings.warn("WARNING: Randomly choosing action for a non human state!")

        _next_states = [_next_n for _next_n in self.graph._graph.successors(_human_state)]
        _did_human_move = False

        # if the human still has moves remaining then he follows the eps strategy else he does not move at all
        if human_can_intervene:
            # rand() return a floating point number between [0, 1)
            if np.random.rand() < epsilon:
                _next_state: tuple = random.choice(_next_states)
            else:
                _next_state: tuple = str_dict[_human_state]

            if _next_state[0][1] != _human_state[0][1]:
                _did_human_move = True
        else:
            # human follows the strategy dictated by the system
            for _n in _next_states:
                if _n[0][1] == _human_state[0][1]:
                    _next_state = _n
                    break

        return _next_state, _did_human_move

    def get_controls_from_str_minigrid(self,
                                       str_dict: Dict[tuple, tuple],
                                       epsilon: float,
                                       max_human_interventions: int = 1,
                                       debug: bool = False) -> List[Tuple[str, ndarray, int]]:
        """
        A helper method to return a list of actions (edge labels) associated with the strategy found

        NOTE: after system node you need to have a human node.
        :param str_dict: The regret minimizing strategy
        :return: A sequence of labels that to be executed by the robot
        """
        if not isinstance(epsilon, float):
            try:
                epsilon = float(epsilon)
                if epsilon > 1:
                    warnings.warn("Please make sure that the value of epsilon is <=1")
            except ValueError:
                print(ValueError)

        if not isinstance(max_human_interventions, int) or max_human_interventions < 0:
            warnings.warn("Please make sure that the max human intervention bound should >= 0")

        _start_state = self.graph.get_initial_states()[0][0]
        _total_human_intervention = _start_state[0][1]
        _accepting_states = self.graph.get_accepting_states()
        _trap_states = self.graph.get_trap_states()

        _absorbing_states = _accepting_states + _trap_states
        _human_interventions: int = 0
        _visited_states = []
        _position_sequence = []

        curr_sys_node = _start_state
        next_env_node = str_dict[curr_sys_node]

        _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
        next_sys_node = self._epsilon_greedy_choose_action(next_env_node,
                                                           str_dict,
                                                           epsilon=epsilon,
                                                           _absoring_states=_absorbing_states,
                                                           human_can_intervene=_can_human_intervene)[0]

        (x, y) = next_sys_node[0][0]
        _human_interventions: int = _total_human_intervention - next_sys_node[0][1]

        next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)

        _visited_states.append(curr_sys_node)
        _visited_states.append(next_sys_node)

        _entered_absorbing_state = False

        while 1:
            _position_sequence.append(next_pos)

            curr_sys_node = next_sys_node
            next_env_node = str_dict[curr_sys_node]

            if curr_sys_node == next_env_node:
                x, y = next_sys_node[0][0]
                _human_interventions: int = _total_human_intervention - next_sys_node[0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(next_sys_node)
                break

            # update the next sys node only if you not transiting to an absorbing state
            if next_env_node not in _absorbing_states and\
                    self.graph.get_state_w_attribute(next_env_node, "player") == "adam":
                _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
                next_sys_node = self._epsilon_greedy_choose_action(next_env_node,
                                                                   str_dict,
                                                                   epsilon=epsilon,
                                                                   _absoring_states=_absorbing_states,
                                                                   human_can_intervene=_can_human_intervene)[0]

            # if transiting to an absorbing state then due to the self transition the next sys node will be the same as
            # the current env node which is an absorbing itself. Technically an absorbing state IS NOT assigned any
            # player.
            else:
                next_sys_node = next_env_node

            if next_sys_node in _visited_states:
                break

            # if you enter a trap/ accepting state then do not add that transition in _pos_sequence
            elif next_sys_node in _absorbing_states:
                _entered_absorbing_state = True
                break
            else:
                x, y = next_sys_node[0][0]
                _human_interventions: int = _total_human_intervention - next_sys_node[0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(next_sys_node)

        if not _entered_absorbing_state:
            _position_sequence.append(next_pos)

        if debug:
            print([_n for _n in _position_sequence])

        return _position_sequence

    def plot_str_from_mcr(self, g_hat: TwoPlayerGraph,
                          str_dict: Dict,
                          only_eve: bool = False,
                          plot: bool = False) -> Dict:
        g_hat.set_edge_attribute('strategy', False)

        if only_eve:
            self._add_strategy_flag_only_eve(g_hat, str_dict)

        else:
            self._add_strategy_flag(g_hat, str_dict)

        _org_str = {}
        # get the org_str
        for _n, _next_n in str_dict.items():
            if isinstance(_n, tuple):
                _n = _n[0]
            if isinstance(_next_n, tuple):
                _next_n = _next_n[0]
            _org_str.update({_n: _next_n})

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, _org_str)

        else:
            self._add_strategy_flag(self.graph, _org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

        return _org_str

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

        org_str = self._from_str_mpg_to_str(str_dict)

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, org_str)

        else:
            self._add_strategy_flag(self.graph, org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

        return org_str
