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
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from ..graph import Graph, graph_factory
from ..graph import TwoPlayerGraph
from ..graph import ProductAutomaton
from ..graph import MiniGrid
from ..payoff import Payoff
from ..mpg_tool import MpgToolBox

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
        self.graph_of_alternatives = None
        self.graph_of_utility = None
        self.b_val: Optional[set] = None

    @property
    def b_val(self):
        return self._b_val

    @b_val.setter
    def b_val(self, value: set):
        self._b_val = value

    def finite_reg_solver_2(self,
                            minigrid_instance,
                            plot: bool = False,
                            plot_only_eve: bool = False,
                            simulate_minigrid: bool = False,
                            epsilon: float = 0,
                            max_human_interventions: int = 5,
                            compute_reg_for_human: bool = False):
        """
        A parent function that computes regret minimizing strategy for the system player for cumulative payoff function.

        In this game, we assume that the weights on the graph represent costs and hence are non-negative. The objective
        of the system player is to minimize its cumulative regret while the env player is trying to maximize the
        cumulative regret.

        Steps:

        1. Add an auxiliary tmp_accp state from the accepting state and the trap state. The edge weight is 0 and W_bar
           respectively. W_bar is equal to : (|V| - 1) x W where W is the max absolute weight
        2. Compute the best cooperative value from each edge
        3. reg^{s, t} = max_t Val^{v}(s, t) - min_t' Val^{v}(s, t)
        4. After computing a reg value associate with a given strategy s, we then fix the str for Env. The strategies
           that env player plays are the memoryless adversarial strategies that maximizes cumulative COST. Then, after
           fixing the strategies for the env player we compute strategies for the sys player that minimizes cumulative
           regret.


        :return:
        """
        # Add auxiliary accepting state
        self.add_common_accepting_state(plot=False)

        # compute cooperative values
        coop_mcr_solver = ValueIteration(self.graph, competitive=False)
        coop_mcr_solver.solve(debug=True, plot=False)
        coop_val_dict = coop_mcr_solver.state_value_dict

        # compute competitive values from each state
        comp_mcr_solver = ValueIteration(self.graph, competitive=True)
        comp_mcr_solver.solve(debug=True, plot=False)
        _env_str_dict = comp_mcr_solver.env_str_dict
        _comp_val_dict = comp_mcr_solver.state_value_dict

        # now we compute reg for each edge(str) as follows: comp(v) - coop(v) where v = sigma(u)
        _sys_reg_val_dict: Dict[Tuple: float] = {}
        _sys_reg_str_dict: Dict[Tuple: float] = {}

        for _s in self.graph._graph.nodes():
            # doing this computation only for sys state
            if self.graph._graph.nodes(data='player')[_s] == 'adam':
                continue

            # get all the outgoing edges
            _min_reg = (math.inf, '')
            for _e in self.graph._graph.out_edges(_s):
                _u = _e[0]
                _v = _e[1]

                _edge_reg_val = _comp_val_dict[_v] - coop_val_dict[_v]
                _sys_reg_val_dict.update({_e: _edge_reg_val})
                if _min_reg[0] > _edge_reg_val:
                    _min_reg = (_edge_reg_val, _v)

            _sys_reg_str_dict.update({_s: _min_reg[1]})

        # combine sys and env str dict
        _reg_str_dict = {**_sys_reg_str_dict, **_env_str_dict}

        # remove edges to the tmp_accp state for ease of plotting
        _curr_tmp_accp = self.graph.get_accepting_states()[0]
        _pre_accp_node = copy.copy(self.graph._graph.predecessors(_curr_tmp_accp))
        _pre_accp: list = []
        for _node in _pre_accp_node:
            self.graph._graph.remove_edge(_node, _curr_tmp_accp)
            self.graph.add_weighted_edges_from([(_node, _node, 0)])
            # our str dict has this term where the original accepting state and the trap state are pointing to the
            # arbitrary tmp_accp state. we need to rectify this

            if _reg_str_dict.get(_node):
                _reg_str_dict[_node] = _node

        if plot:
            self.plot_str_for_cumulative_reg(game_venue=self.graph,
                                             str_dict=_reg_str_dict,
                                             only_eve=plot_only_eve,
                                             plot=plot)

        if simulate_minigrid:
            # get the regret value of the game
            _init_state = self.graph.get_initial_states()[0][0]
            _game_reg_value: float = math.inf
            for _e in self.graph._graph.out_edges(_init_state):
                if _sys_reg_val_dict.get(_e):
                    _game_reg_value = _sys_reg_val_dict[_e]
                    break

            self.graph.add_accepting_state("accept_all")
            self.graph.remove_state_attr("tmp_accp", "accepting")
            if minigrid_instance is None:
                warnings.warn("Please provide a Minigrid instance to simulate!. Exiting program")
                sys.exit(-1)

            _controls = self.get_controls_from_str_minigrid(str_dict=_reg_str_dict,
                                                            epsilon=epsilon,
                                                            max_human_interventions=max_human_interventions)

            minigrid_instance.execute_str(_controls=(_game_reg_value, _controls))

    def finite_reg_solver_1(self,
                            minigrid_instance,
                            plot: bool = False,
                            plot_only_eve: bool = False,
                            simulate_minigrid: bool = False,
                            epsilon: float = 0,
                            max_human_interventions: int = 5,
                            compute_reg_for_human: bool = False):
        """
        A parent function that computes regret minimizing strategy for the system player for cumulative payoff function.

        In this game, we assume that the weights on the graph represent costs and hence are non-negative. The objective
        of the system player is to minimize its cumulative regret while the env player is trying to maximize the
        cumulative regret.

        Steps:

        1. Add an auxiliary tmp_accp state from the accepting state and the trap state. The edge weight is 0 and W_bar
           respectively. W_bar is equal to : (|V| - 1) x W where W is the max absolute weight
        2. Compute the best cooperative value from each edge
        3. reg^{s, t} = max_t Val^{v}(s, t) - cVal^{v}
        4. After computing a reg value associated with a given strategy s, we then fix the str for Env. The strategies
           that env player plays are the memoryless adversarial strategies that maximizes cumulative COST. Then, after
           fixing the strategies for the env player we compute strategies for the sys player that minimizes cumulative
           regret.


        :return:
        """
        # Add auxiliary accepting state
        self.add_common_accepting_state(plot=False)

        # compute cooperative str
        coop_mcr_solver = ValueIteration(self.graph, competitive=False)
        coop_mcr_solver.solve(debug=True, plot=False)
        coop_val_dict = coop_mcr_solver.state_value_dict

        # compute competitive values from each state
        comp_mcr_solver = ValueIteration(self.graph, competitive=True)
        comp_mcr_solver.solve(debug=True, plot=False)
        _comp_str_dict = comp_mcr_solver.str_dict
        _comp_val_dict = comp_mcr_solver.state_value_dict

        # compute reg value associated with each state
        _reg_val_dict: Dict[Tuple: float] = {}

        for _s in self.graph._graph.nodes:

            _reg_val: float = _comp_val_dict[_s] - coop_val_dict[_s]
            _reg_val_dict.update({_s: _reg_val})

        # remove edges to the tmp_accp state for ease of plotting
        _curr_tmp_accp = self.graph.get_accepting_states()[0]
        _pre_accp_node = copy.copy(self.graph._graph.predecessors(_curr_tmp_accp))
        _pre_accp : list = []
        for _node in _pre_accp_node:
            self.graph._graph.remove_edge(_node, _curr_tmp_accp)
            self.graph.add_weighted_edges_from([(_node, _node, 0)])
            # our str dict has this term where the original accepting state and the trap state are pointing to the
            # arbitrary tmp_accp state. we need to rectify this

            if _comp_str_dict.get(_node):
                _comp_str_dict[_node] = _node

        if plot:
            self.plot_str_for_cumulative_reg(game_venue=self.graph,
                                             str_dict=_comp_str_dict,
                                             only_eve=plot_only_eve,
                                             plot=plot)

        if simulate_minigrid:
            # get the regret value of the game
            _init_state = self.graph.get_initial_states()[0][0]
            _game_reg_value: float = _reg_val_dict.get(_init_state)

            self.graph.add_accepting_state("accept_all")
            self.graph.remove_state_attr("tmp_accp", "accepting")
            if minigrid_instance is None:
                warnings.warn("Please provide a Minigrid instance to simulate!. Exiting program")
                sys.exit(-1)

            _controls = self.get_controls_from_str_minigrid(str_dict=_comp_str_dict,
                                                            epsilon=epsilon,
                                                            max_human_interventions=max_human_interventions)

            minigrid_instance.execute_str(_controls=(_game_reg_value, _controls))

    def add_common_accepting_state(self, plot: bool = False):
        """
        A helper method that adds a auxiliary accepting state from the current accepting state as well as the trap state
        to the new accepting state. This helper methos is called by the finite_reg_solver method to augment the existing
        graph as follows:

        trap--W_bar --> new_accp (self-loop weight 0)
                 /^
                /
               0
              /
             /
            /
        Acc

        W_bar : Highest payoff any state could ever achieve when playing total-payoff game in cooperative setting,
        assuming strategies to be non-cylic, will be equal to (|V| -1)W where W is the max absolute weight.

        Now we remove Acc as the accepting state and add new_accp as the new accepting state and initialize this state
        to 0 in the value iteration algorithm - essentially eliminating a trap region.

        :return:
        """

        old_accp_states = self.graph.get_accepting_states()
        trap_states = self.graph.get_trap_states()
        _num_of_nodes = len(list(self.graph._graph.nodes))
        _W = abs(self.graph.get_max_weight())
        w_bar = ((_num_of_nodes - 1) * _W)

        # remove self-loops of accepting states and add edge from that state to the new accepting state with edge
        # weight 0
        for _accp in old_accp_states:
            self.graph._graph.remove_edge(_accp, _accp)
            self.graph.remove_state_attr(_accp, "accepting")
            self.graph.add_state_attribute(_accp, "player", "eve")
            self.graph.add_weighted_edges_from([(_accp, 'tmp_accp', 0)])

        # remove self-loops of trap states and add edge from that state to the new accepting state with edge weight
        # w_bar
        for _trap in trap_states:
            self.graph._graph.remove_edge(_trap, _trap)
            self.graph.add_state_attribute(_trap, "player", "eve")
            self.graph.add_weighted_edges_from([(_trap, 'tmp_accp', w_bar)])

        self.graph.add_weighted_edges_from([('tmp_accp', 'tmp_accp', 0)])
        self.graph.add_accepting_state('tmp_accp')
        self.graph.add_state_attribute('tmp_accp', "player", "eve")

        if plot:
            self.graph.plot_graph()

    def edge_weighted_arena_finite_reg_solver(self,
                                              purge_states: bool = True,
                                              plot: bool = False):
        """
        A function that computes a Regret Minimizing strategy by constructing the Graph of Utility G'. This graph (G')
        is of the form Target weighted arena (TWA).  The utility information is added to the nodes of G to construct
        the nodes of G'. The utility information added to the nodes is uniquely determined by the path used to reach
        the current  position.

        We then call the TWA regret solver code to compute regret minimizing strategies for the Sys player.

        Input: An Edge Weighted arena with all the edges having weight >= 0. Remember that we cannot have a zero payoff
        loop.

        Output: The strategies on G' (graph of utility) are finite-memory strategies whose memory is the best
        alternative seen so far. When we map back the strategies to G, this gives a finite memory strategy whose memory
        is the utility of the current play up to the Bounded value = W * |S| AND the best alternative seen so far.

        :return:
        """
        # # Add auxiliary accepting state
        # self.add_common_accepting_state(plot=False)

        # construct a TWA given the graph
        self.graph_of_utility = self._construct_graph_of_utility()

        # helper method to remove the state that cannot reached from the initial state of G'
        if purge_states:
            self._remove_non_reachable_states(self.graph_of_utility)

        if plot:
            self.graph_of_utility.plot_graph()

        # Compute strs on this new TWA Graph
        self.target_weighted_arena_finite_reg_solver(twa_graph=self.graph_of_utility,
                                                     debug=False,
                                                     plot_w_vals=True,
                                                     plot_only_eve=False,
                                                     plot=False)

    def target_weighted_arena_finite_reg_solver(self,
                                                twa_graph: TwoPlayerGraph,
                                                debug: bool = False,
                                                purge_states: bool = True,
                                                plot_w_vals: bool = False,
                                                plot_only_eve: bool = False,
                                                plot: bool = False):
        """
        A function to compute a Regret Minimizing strategy by constructing the Graph of best alternative G'.
        Please refer to  arXiv:1002.1456v3 Section 2 For the theory.

        We first compute G' and solve a minmax game. This gives us a memoryless strategy that achieves the minimal
        regret in the graph of best alternatives. We map these strategies on to G, this gives us a finite memory
        strategy whose is exactly the best alternative seen along the current finite play. Therefore, the memory is
        bounded by the number of best alternatives which is bounded by the number of leaf nodes in the TWA.

        Input: A Target weighted arena with all the edges that do not transit to a target (leaf node) have zero edge
        weight and edges that transit to the same leaf node have the same edge weight.

        Output: A finite-memory regret minimizing strategy for the Min/Sys player.

        :return:
        """
        # compute the best alternative from each edge for cumulative payoff
        _best_alternate_values: Dict = self._get_best_alternatives_dict(twa_graph)

        # construct graph of best alternatives (G')
        self.graph_of_alternatives =\
            self._construct_graph_of_best_alternatives(twa_game=twa_graph,
                                                       best_alt_values_dict=_best_alternate_values)

        # for all the edge that transit to a target state we need to compute the regret associate with that
        self._compute_reg_for_edge_to_target_nodes(game=self.graph_of_alternatives)

        # purge nodes that are not reachable form the init state
        if purge_states:
            self._remove_non_reachable_states(self.graph_of_alternatives)

        # play minmax game to compute regret minimizing strategy
        minmax_mcr_solver = ValueIteration(self.graph_of_alternatives, competitive=True)
        minmax_mcr_solver.solve(debug=True, plot=plot_w_vals)
        _comp_str_dict = minmax_mcr_solver.str_dict
        _comp_val_dict = minmax_mcr_solver.state_value_dict

        if plot_only_eve:
            self.plot_str_for_cumulative_reg(game_venue=self.graph_of_alternatives,
                                             str_dict=_comp_str_dict,
                                             only_eve=plot_only_eve,
                                             plot=plot)

    def _construct_graph_of_utility(self):
        """
        A function to construct the graph of utility given an Edge Weighted Arena (EWA).

        We populate all the states with all the possible utilities. A position may be reachable by several paths,
        therefore it will be duplicated as many times as there are different path utilities. This duplication is
        bounded the value B = 2 * W * |S|. Refer to Lemma 4 of the paper for more details.

        Constructing G' (Graph of utility (TWA)):

        S' = S x [B]; Where B is the an positive integer defined as above
        An edge between two states (s, u),(s, u') exists iff s to s' is a valid edge in G and u' = u + w(s, s')

        c' = S' ∩ [C1 x [B]] are the target states in G'. THe edge weight of edges transiting to a target state (s, u)
        is u. All the other edges have an edge weight 0. (Remember this is a TWA with non-zero edge weights on edges
        transiting to the target states.)

        :return:
        """

        # get the max bound on the value of strategies
        _max_weight: Optional[int, float] = self.graph.get_max_weight()

        if isinstance(_max_weight, float):
            warnings.warn("Max weight is of type float. For TWA Construction max weight should be a integer")

        _max_bounded_str_value = 2 * _max_weight * len(self.graph._graph.nodes)

        _graph_of_utls = graph_factory.get("TwoPlayerGraph",
                                           graph_name="graph_of_utls_TWA",
                                           config_yaml="/config/graph_of_ults_TWA",
                                           save_flag=True,
                                           pre_built=False,
                                           from_file=False,
                                           plot=False)

        # get initial states
        _init_state = self.graph.get_initial_states()[0][0]

        # construct nodes
        for _s in self.graph._graph.nodes():
            for _u in range(_max_bounded_str_value + 1):
                _org_state_attrs = self.graph._graph.nodes[_s]
                _new_state = (_s, _u)
                _graph_of_utls.add_state(_new_state, **_org_state_attrs)
                _graph_of_utls._graph.nodes[_new_state]['accepting'] = False
                _graph_of_utls._graph.nodes[_new_state]['init'] = False

                if _s == _init_state and _u == 0:
                    _graph_of_utls._graph.nodes[_new_state]['init'] = True

        # construct edges
        for _s in self.graph._graph.nodes():
            for _u in range(_max_bounded_str_value + 1):
                _curr_state = (_s, _u)

                # get the org neighbours of the _s in the org graph
                for _org_succ in self.graph._graph.successors(_s):
                    # the edge weight between these two, add _u to this edge weight to get _u'. Add this edge to G'
                    _org_edge_w: int = self.graph.get_edge_attributes(_s, _org_succ, "weight")

                    if not isinstance(_org_edge_w, int):
                        warnings.warn(f"Got an invalid edge weight type. The edge weight for edge {_s} -> {_org_succ}"
                                      f" is of type {type(_org_edge_w)}")

                    _next_u = _u + _org_edge_w

                    _succ_state = (_org_succ, _next_u)

                    if not _graph_of_utls._graph.has_node(_succ_state):
                        warnings.warn(f"Trying to add a new node {_succ_state} to the graph of utility."
                                      f"This should not happen. Check your construction code")
                        continue

                    _org_edge_attrs = self.graph._graph.edges[_s, _org_succ, 0]
                    _graph_of_utls.add_edge(u=_curr_state,
                                            v=_succ_state,
                                            **_org_edge_attrs)

                    _graph_of_utls._graph[_curr_state][_succ_state][0]['weight'] = 0

        # construct target states
        _accp_states: list = self.graph.get_accepting_states()

        for _accp_s in _accp_states:
            for _u in range(_max_bounded_str_value + 1):
                _new_accp_s = (_accp_s, _u)

                if not _graph_of_utls._graph.has_node(_new_accp_s):
                    warnings.warn(f"Trying to add a new accepting node {_new_accp_s} to the graph of best alternatives."
                                  f"This should not happen. Check your construction code")

                _graph_of_utls.add_accepting_state(_new_accp_s)

                # also we need to add edge weight to target states.
                for _pre_s in _graph_of_utls._graph.predecessors(_new_accp_s):
                    if _pre_s == _new_accp_s:
                        continue
                    _graph_of_utls._graph[_pre_s][_new_accp_s][0]['weight'] = _u

        return _graph_of_utls

    def _construct_graph_of_best_alternatives(self,
                                              twa_game: TwoPlayerGraph,
                                              best_alt_values_dict: Dict) -> TwoPlayerGraph:
        """
        A function that construct the graph of best alternative.

        Constructing G' (Graph of best alternative):

        S' = S x ([W] U {+inf}) ; W = Maximum weight in the Graph. and [W] = [0, 1, 2, .... W]
        An edge between two states (s, b), (s, b') exists iff s to s' is a valid edge in G and
            b' = b if s belong to the MAX/ Env player
            b' = min(b, ba(s, s')) if s belongs to MIN/ Sys player

        ba(s, s'): IS the best alternative value you can secure if you take some other edge from s.
            ba(s, s') = +inf if s belongs to MAX/Env player
            ba(s, s') = min of all the edges and +inf if there are no alternate edges

        C' = S' ∩ (C1 x [W]) are the target states in G' and the edge weight on G' is the edge weight on the original
        graph.

        :return:
        """

        _graph_of_alts = graph_factory.get("TwoPlayerGraph",
                                           graph_name="graph_of_alts_TWA",
                                           config_yaml="/config/graph_of_alts_TWA",
                                           save_flag=True,
                                           pre_built=False,
                                           from_file=False,
                                           plot=False)

        # get the max weight and create a set([W] U {+inf}) = [0, 1, 2, ... W, +inf]
        _max_weight: Optional[int, float] = twa_game.get_max_weight()

        # get initial states
        _init_state = twa_game.get_initial_states()[0][0]

        if isinstance(_max_weight, float):
            warnings.warn("Max weight is of type float. For TWA Construction max weight should be a integer")

        _possible_best_alt_values = [i for i in range(_max_weight + 1)]
        _possible_best_alt_values.append(math.inf)

        # construct nodes
        for _s in twa_game._graph.nodes():
            for _best_alt in _possible_best_alt_values:
                # get original state attributes
                _org_state_attrs = twa_game._graph.nodes[_s]
                _new_state = (_s, _best_alt)
                _graph_of_alts.add_state(_new_state, **_org_state_attrs)
                _graph_of_alts._graph.nodes[_new_state]['accepting'] = False
                _graph_of_alts._graph.nodes[_new_state]['init'] = False
                # only the original initial state with +inf as best alternative should be assigned the init state attr
                if _s == _init_state and _best_alt == math.inf:
                    _graph_of_alts._graph.nodes[_new_state]['init'] = True

        # add valid transition
        for _s in twa_game._graph.nodes():
            for _best_alt in _possible_best_alt_values:
                _curr_state = (_s, _best_alt)
                if twa_game.get_state_w_attribute(_s, "player") == "adam":
                    # get successors of the MAX/Env player state in the original graph and add edge to the successor
                    # with same best alternate values
                    for _org_succ in twa_game._graph.successors(_s):
                        _succ = (_org_succ, _best_alt)

                        if not _graph_of_alts._graph.has_node(_succ):
                            warnings.warn(f"Trying to add a new node {_succ} to the graph of best alternatives."
                                          f"This should not happen. Check your construction code")

                        _org_edge_attrs = twa_game._graph.edges[_s, _org_succ, 0]
                        _graph_of_alts.add_edge(u=_curr_state,
                                                v=_succ,
                                                **_org_edge_attrs)

                elif twa_game.get_state_w_attribute(_s, "player") == "eve":
                    # get the successors of the MIN/Sys player state in the original graph and edge to the successor
                    # who have satisfy min(b', ba(s, s'))
                    for _org_succ in twa_game._graph.successors(_s):
                        _ba = best_alt_values_dict.get((_s, _org_succ))
                        _next_state_ba_value = min(_best_alt, _ba)

                        _succ = (_org_succ, _next_state_ba_value)
                        if not _graph_of_alts._graph.has_node(_succ):
                            warnings.warn(f"Trying to add a new node {_succ} to the graph of best alternatives."
                                          f"This should not happen. Check your construction code")

                        _org_edge_attrs = twa_game._graph.edges[_s, _org_succ, 0]
                        _graph_of_alts.add_edge(u=_curr_state,
                                                v=_succ,
                                                **_org_edge_attrs)
                else:
                    warnings.warn(f"Encountered a state {_s} with an invalid player attribute")

        # add accepting state attribute
        _accp_states: list = twa_game.get_accepting_states()

        for _accp_s in _accp_states:
            for _ba_val in range(_max_weight + 1):
                _new_accp_s = (_accp_s, _ba_val)

                if not _graph_of_alts._graph.has_node(_new_accp_s):
                    warnings.warn(f"Trying to add a new accepting node {_new_accp_s} to the graph of best alternatives."
                                  f"This should not happen. Check your construction code")

                _graph_of_alts.add_accepting_state(_new_accp_s)

        return _graph_of_alts

    def _remove_non_reachable_states(self, game: Graph):
        """
        A helper method that removes all the states that are not reachable from the initial state. This method is
        called by the edge weighted are reg solver method to trim states and reduce the size of the graph

        :param game:
        :return:
        """
        # get the initial state
        _init_state = game.get_initial_states()[0][0]
        _org_node_set: set = set(game._graph.nodes())

        _visited = set()

        # add the initial state to the visit stack
        def dfs(visited: set, node):
            if node not in visited:
                visited.add(node)
                for _neighbour in game._graph.successors(node):
                    if _neighbour == node:
                        continue
                    dfs(visited, _neighbour)

            return visited

        _valid_states = dfs(_visited, _init_state)

        _nodes_to_be_purged = _org_node_set - _valid_states
        game._graph.remove_nodes_from(_nodes_to_be_purged)

    def _compute_reg_for_edge_to_target_nodes(self, game: Graph):
        """
        A helper function that compute the regret value for edges that transit to a target nodes in the graph of best
        alternatives as per function

        v'(s') = edge_weight_to(s) - min(edge_weight_to(s), b)

        State s is of the form: s' = (s, b) where s is the original state in the graph and b is the best alternative
        value along the path taken to reach (s, b).
        :return:
        """
        # get all the accepting states
        _accp_s = game.get_accepting_states()

        # compute reg value as per the doc string
        for _target in _accp_s:
            for _pre_s in game._graph.predecessors(_target):
                _ba_val: int = _target[1]
                org_edge: int = game.get_edge_attributes(_pre_s, _target, "weight")
                _reg_value: int = org_edge - min(org_edge, _ba_val)

                game._graph[_pre_s][_target][0]['weight'] = _reg_value

    def _get_best_alternatives_dict(self, two_player_game: TwoPlayerGraph) -> Dict:
        """
        A function that computes the best alternate (ba) value for each edge in the graph.

        If a state belongs to Adam :- ba = +inf
        if a state belongs to Eve :- ba(s, s') = minimum of all the cooperative values for successors s'' s.t.
         s'' not equal to s'.  If there is no alternate edge to choose from, then ba(s, s') = +inf
        :return:
        """

        # pre-compute cooperative values form each state
        coop_mcr_solver = ValueIteration(two_player_game, competitive=False)
        coop_mcr_solver.solve(debug=False, plot=False)
        coop_val_dict = coop_mcr_solver.state_value_dict

        _best_alternate_values: Dict[Optional[tuple], Optional[int, float]] = defaultdict(lambda: -1)

        for _e in two_player_game._graph.edges():
            _u = _e[0]
            _v = _e[1]

            if two_player_game.get_state_w_attribute(_u, "player") == "adam":
                _best_alternate_values.update({_e: math.inf})

            elif two_player_game.get_state_w_attribute(_u, "player") == "eve":
                _min_coop_val = math.inf
                for _succ in two_player_game._graph.successors(_u):
                    if _succ == _v:
                        continue

                    _curr_edge_weight = two_player_game.get_edge_attributes(_u, _succ, "weight")

                    _curr_edge_coop_val: Optional[int, float] = _curr_edge_weight + coop_val_dict.get(_succ)
                    if _curr_edge_weight < _min_coop_val:
                        _min_coop_val = _curr_edge_coop_val

                _best_alternate_values.update({_e: _min_coop_val})

            else:
                warnings.warn(f"Encountered a state {_u} with an invalid player attribute")

        return _best_alternate_values

    def infinite_reg_solver(self,
                           minigrid_instance,
                           plot: bool = False,
                           plot_only_eve: bool = False,
                           simulate_minigrid: bool = False,
                           go_fast: bool = True,
                           finite: bool = False,
                           epsilon: float = 0,
                           max_human_interventions: int = 5):
        """
        A method to compute payoff when using a type of infinite payoff. The weights associated with the game are costs
        and are non-negative.

        :param minigrid_instance:
        :param plot:
        :param plot_only_eve:
        :param simulate_minigrid:
        :param epsilon:
        :param max_human_interventions:
        :return:
        """

        # compute w_prime
        w_prime = self.compute_W_prime(go_fast=go_fast, debug=False)

        g_hat = self.construct_g_hat(w_prime, game=None, finite=finite, debug=True,
                                     plot=False)

        mpg_g_hat_handle = MpgToolBox(g_hat, "g_hat")

        reg_dict, reg_val = mpg_g_hat_handle.compute_reg_val(go_fast=True, debug=False)
        # g_hat.plot_graph()
        org_str = self.plot_str_from_mgp(g_hat, reg_dict, only_eve=plot_only_eve, plot=plot)

        if minigrid_instance is not None:
            controls = self.get_controls_from_str_minigrid(org_str,
                                                           epsilon=epsilon,
                                                           max_human_interventions=max_human_interventions)
            minigrid_instance.execute_str(_controls=(reg_val, controls))

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

    def plot_str_for_cumulative_reg(self,
                                    game_venue: TwoPlayerGraph,
                                    str_dict: Dict,
                                    only_eve: bool = False,
                                    plot: bool = False):
        """
        A helper method that add a strategy flag to very edge in the original game. Then it iterates through the
        strategy dict and sets the edges for both the sys player and env player in the strategy dict as True.

        The plot_graph() function colors edges that have this strategy as red.
        :return:
        """

        self.graph.set_edge_attribute('strategy', False)

        if only_eve:
            self._add_strategy_flag_only_eve(game_venue, str_dict)
        else:
            self._add_strategy_flag(game_venue, str_dict)

        if plot:
            game_venue.plot_graph()


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
