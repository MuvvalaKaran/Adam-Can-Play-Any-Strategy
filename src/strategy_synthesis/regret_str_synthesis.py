import math
import copy
import multiprocessing
import warnings
import random
import sys
import operator

import networkx as nx
import numpy as np
from numpy import ndarray
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from ..graph import Graph, graph_factory
from ..graph import TwoPlayerGraph
from ..graph import ProductAutomaton
from ..helper_methods import deprecated

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
    """
    def __init__(self,
                 graph: TwoPlayerGraph) -> 'RegretMinimizationStrategySynthesis':
        self.graph = graph
        self.graph_of_alternatives = None
        self.graph_of_utility = None

    @deprecated
    def finite_reg_solver_2(self,
                            plot: bool = False,
                            plot_only_eve: bool = False):
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

        Note: This implementaton is flawed will be updated in future commits.
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

    def finite_reg_solver_1(self,
                            plot: bool = False,
                            plot_only_eve: bool = False):
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
        coop_mcr_solver.cooperative_solver(debug=False, plot=False)
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

    def pure_games_solver(self,
                          minigrid_instance,
                          cooperative: bool,
                          plot: bool = False,
                          plot_only_eve: bool = False,
                          simulate_minigrid: bool = False,
                          epsilon: float = 0,
                          max_human_interventions: int = 5,
                          compute_reg_for_human: bool = False,
                          integrate_accepting: bool = False):
        """
        A parent function that computes a strategy for the system player under PURE ADVERSARIAL game.

        :return:
        """
        # Add auxiliary accepting state
        if integrate_accepting:
            self.add_common_accepting_state(plot=False)

        # # compute cooperative str
        if cooperative:
            coop_mcr_solver = ValueIteration(self.graph, competitive=False, int_val=False)
            coop_mcr_solver.cooperative_solver(debug=True, plot=plot)
            _comp_str_dict = coop_mcr_solver.str_dict
            _comp_val_dict = coop_mcr_solver.state_value_dict
        # compute competitive values from each state
        else:
            comp_mcr_solver = ValueIteration(self.graph, competitive=True, int_val=False)
            comp_mcr_solver.solve(debug=True, plot=plot)
            _comp_str_dict = comp_mcr_solver.str_dict
            _comp_val_dict = comp_mcr_solver.state_value_dict

        if simulate_minigrid:
            # get the regret value of the game
            _init_state = self.graph.get_initial_states()[0][0]
            _game_reg_value: float = _comp_val_dict.get(_init_state)

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
        # w_bar = ((_num_of_nodes - 1) * _W)
        w_bar = 0

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
                                              plot: bool = False) -> Tuple[Dict, Dict]:
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
        # compute the minmax value of the game
        comp_mcr_solver = ValueIteration(self.graph, competitive=True)
        comp_mcr_solver.solve(debug=False, plot=False)

        # init state value
        _init_state = self.graph.get_initial_states()[0][0]
        min_max_value = comp_mcr_solver.state_value_dict[_init_state] * 1.2
        # min_max_value = comp_mcr_solver.state_value_dict[_init_state].item()
        min_max_value = math.ceil(min_max_value)
        # min_max_value = 6
        # construct a TWA given the graph
        self.graph_of_utility = self._construct_graph_of_utility(min_max_value)

        print(f"#nodes in the graph of utility before pruning:{len(self.graph_of_utility._graph.nodes())}")
        print(f"#edges in the graph of utility before pruning:{len(self.graph_of_utility._graph.edges())}")

        # helper method to remove the state that cannot reached from the initial state of G'
        if purge_states:
            self._remove_non_reachable_states(self.graph_of_utility)

        if plot:
            self.graph_of_utility.plot_graph()

        print(f"#nodes in the graph of utility after pruning:{len(self.graph_of_utility._graph.nodes())}")
        print(f"#edges in the graph of utility after pruning:{len(self.graph_of_utility._graph.edges())}")

        # Compute strs on this new TWA Graph
        reg_strs, reg_vals = self.target_weighted_arena_finite_reg_solver(twa_graph=self.graph_of_utility,
                                                                          debug=True,
                                                                          plot_w_vals=False,
                                                                          plot_only_eve=False,
                                                                          plot=False)

        return reg_strs, reg_vals

    def target_weighted_arena_finite_reg_solver(self,
                                                twa_graph: TwoPlayerGraph,
                                                debug: bool = False,
                                                purge_states: bool = True,
                                                plot_w_vals: bool = False,
                                                plot_only_eve: bool = False,
                                                plot: bool = False) -> Tuple[Dict, float]:
        """
        A function to compute a Regret Minimizing strategy by constructing the Graph of best alternative G'.
        Please refer to  arXiv:1002.1456v3 Section 2 For the theory.

        We first compute G' and solve a minmax game. This gives us a memoryless strategy that achieves the minimal
        regret in the graph of best alternatives. We map these strategies on to G, this gives us a finite memory
        strategy whose memory is exactly the best alternative seen along the current finite play. Therefore, the memory
        is bounded by the number of best alternatives which is bounded by the number of leaf nodes in the TWA.

        Input: A Target weighted arena with all the edges that do not transit to a target (leaf node) have zero edge
        weight and edges that transit to the same leaf node have the same edge weight.

        Output: A finite-memory regret minimizing strategy for the Min/Sys player.

        :return:
        """
        # compute the best alternative from each edge for cumulative payoff
        _best_alternate_values: Dict = self._get_best_alternatives_dict(twa_graph)

        # construct graph of best alternatives (G')
        self.graph_of_alternatives =\
            self._new_construct_graph_of_best_alternatives(twa_game=twa_graph,
                                                           best_alt_values_dict=_best_alternate_values)

        if debug:
            print(f"#nodes in the graph of alternative before pruning :{len(self.graph_of_alternatives._graph.nodes())}")
            print(f"#edges in the graph of alternative before pruning :{len(self.graph_of_alternatives._graph.edges())}")

        # for all the edge that transit to a target state we need to compute the regret associate with that
        self._compute_reg_for_edge_to_target_nodes(game=self.graph_of_alternatives)

        # purge nodes that are not reachable form the init state
        if purge_states:
            self._remove_non_reachable_states(self.graph_of_alternatives)

        if debug:
            print(f"#nodes in the graph of alternative after pruning :{len(self.graph_of_alternatives._graph.nodes())}")
            print(f"#edges in the graph of alternative after pruning :{len(self.graph_of_alternatives._graph.edges())}")

        # play minmax game to compute regret minimizing strategy
        minmax_mcr_solver = ValueIteration(self.graph_of_alternatives, competitive=True)
        minmax_mcr_solver.solve(debug=True, plot=plot_w_vals)
        _comp_str_dict = minmax_mcr_solver.str_dict
        _comp_val_dict = minmax_mcr_solver.state_value_dict
        _init_state = self.graph_of_alternatives.get_initial_states()[0][0]
        _game_reg_value: float = _comp_val_dict.get(_init_state)

        if plot_only_eve:
            self.plot_str_for_cumulative_reg(game_venue=self.graph_of_alternatives,
                                             str_dict=_comp_str_dict,
                                             only_eve=plot_only_eve,
                                             plot=plot)

        return _comp_str_dict, _comp_val_dict

    def _construct_graph_of_utility(self, min_max_val):
        """
        A function to construct the graph of utility given an Edge Weighted Arena (EWA).

        We populate all the states with all the possible utilities. A position may be reachable by several paths,
        therefore it will be duplicated as many times as there are different path utilities. This duplication is
        bounded the value B = 2 * W * |S|. Refer to Lemma 4 of the paper for more details.

        Constructing G' (Graph of utility (TWA)):

        S' = S x [B]; Where B is the an positive integer defined as above
        An edge between two states (s, u),(s, u') exists iff s to s' is a valid edge in G and u' = u + w(s, s')

        C' = S' ∩ [C1 x [B]] are the target states in G'. THe edge weight of edges transiting to a target state (s, u)
        is u. All the other edges have an edge weight 0. (Remember this is a TWA with non-zero edge weights on edges
        transiting to the target states.)

        :return:
        """

        _max_bounded_str_value = min_max_val

        _graph_of_utls = TwoPlayerGraph("graph_of_utls_TWA", "config/graph_of_ults_TWA", True)

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

        # manually add a sink state with player attribute adam
        # we add sink as accepting so as not not backpropagate inf in coop value plays
        _graph_of_utls.add_state("vT", accepting=False, init=False, player="eve")

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
                        if _next_u <= _max_bounded_str_value:
                            warnings.warn(f"Trying to add a new node {_succ_state} to the graph of utility."
                                          f"This should not happen. Check your construction code")
                            continue

                    # if the next state is within the bounds then, add that state to the graph of utility with edge
                    # weight 0
                    if _next_u <= _max_bounded_str_value:
                        _org_edge_attrs = self.graph._graph.edges[_s, _org_succ, 0]
                        _graph_of_utls.add_edge(u=_curr_state,
                                                v=_succ_state,
                                                **_org_edge_attrs)

                        _graph_of_utls._graph[_curr_state][_succ_state][0]['weight'] = 0

                    # if _next_u > _max_bounded_str_value then add an edge from the current state to a sink state
                    if _next_u > _max_bounded_str_value:
                        # add a self loop with edge weight 0 if it already does not exists
                        _succ_state = "vT"
                        if not _graph_of_utls._graph.has_edge(_curr_state, _succ_state):
                            _graph_of_utls.add_edge(u=_curr_state, v=_succ_state, weight=_max_bounded_str_value)
                        # if not _graph_of_utls._graph.has_edge(_succ_state, _succ_state):
                        #     _graph_of_utls.add_edge(u=_succ_state,
                        #                             v=_succ_state,
                        #                             weight=0)

        # manually add a self-loop to the terminal state because every states should have atleast one out-going edges
        # the self-loop will have edge weight 0.
        _graph_of_utls.add_edge(u="vT",
                                v="vT",
                                weight=0)

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

    def _new_construct_graph_of_best_alternatives(self,
                                                 twa_game: TwoPlayerGraph,
                                                 best_alt_values_dict: Dict) -> TwoPlayerGraph:

        _graph_of_alts = TwoPlayerGraph("graph_of_alts_TWA", "config/graph_of_alts_TWA", True)

        # get the set of best alternatives - will already include inf
        _best_alt_set = set(best_alt_values_dict.values())

        # get initial states
        _init_state = twa_game.get_initial_states()[0][0]

        # construct nodes
        for _s in twa_game._graph.nodes():
            for _best_alt in _best_alt_set:
                # get original state attributes
                _org_state_attrs = twa_game._graph.nodes[_s]
                # if _s is the init state then only make the inf copy of it
                if _s == _init_state:
                    _new_state = (_s, math.inf)
                    _graph_of_alts.add_state(_new_state, **_org_state_attrs)
                    _graph_of_alts._graph.nodes[_new_state]['accepting'] = False
                    _graph_of_alts._graph.nodes[_new_state]['init'] = True
                else:
                    _new_state = (_s, _best_alt)
                    _graph_of_alts.add_state(_new_state, **_org_state_attrs)
                    _graph_of_alts._graph.nodes[_new_state]['accepting'] = False
                    _graph_of_alts._graph.nodes[_new_state]['init'] = False

        # add valid transition
        for _s in twa_game._graph.nodes():
            for _best_alt in _best_alt_set:
                # as we only make init state with inf value, we skip for other iterations
                if _s == _init_state and _best_alt != math.inf:
                    continue
                _curr_state = (_s, _best_alt)
                if twa_game.get_state_w_attribute(_s, "player") == "adam":
                    # get successors of the MAX/Env player state in the original graph and add edge to the successor
                    # with same best alternate values
                    for _org_succ in twa_game._graph.successors(_s):
                        _succ = (_org_succ, _best_alt)

                        # if not _graph_of_alts._graph.has_node(_succ):
                        #     warnings.warn(f"Trying to add a new node {_succ} to the graph of best alternatives."
                        #                   f"This should not happen. Check your construction code")

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
                        # if not _graph_of_alts._graph.has_node(_succ):
                        #     warnings.warn(f"Trying to add a new node {_succ} to the graph of best alternatives."
                        #                   f"This should not happen. Check your construction code")

                        _org_edge_attrs = twa_game._graph.edges[_s, _org_succ, 0]
                        _graph_of_alts.add_edge(u=_curr_state,
                                                v=_succ,
                                                **_org_edge_attrs)
                # else:
                #     warnings.warn(f"Encountered a state {_s} with an invalid player attribute")

        _accp_states: list = twa_game.get_accepting_states()

        for _accp_s in _accp_states:
            for _ba_val in _best_alt_set:
                _new_accp_s = (_accp_s, _ba_val)

                if not _graph_of_alts._graph.has_node(_new_accp_s):
                    warnings.warn(
                        f"Trying to add a new accepting node {_new_accp_s} to the graph of best alternatives."
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
        print("Starting purging nodes")
        # get the initial state
        _init_state = game.get_initial_states()[0][0]
        _org_node_set: set = set(game._graph.nodes())

        stack = deque()
        path: set = set()

        stack.append(_init_state)
        while stack:
            vertex = stack.pop()
            if vertex in path:
                continue
            path.add(vertex)
            for _neighbour in game._graph.successors(vertex):
                stack.append(_neighbour)

        _valid_states = path

        _nodes_to_be_purged = _org_node_set - _valid_states
        game._graph.remove_nodes_from(_nodes_to_be_purged)
        print("Done purging nodes")

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
        coop_mcr_solver.cooperative_solver(debug=False, plot=False)
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

            # else:
            #     warnings.warn(f"Encountered a state {_u} with an invalid player attribute")

        return _best_alternate_values

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

    def _epsilon_greedy_finite_choose_action(self,
                                             graph,
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

        if graph.get_state_w_attribute(_human_state, "player") != "adam":
            warnings.warn("WARNING: Randomly choosing action for a non human state!")

        _next_states = [_next_n for _next_n in graph._graph.successors(_human_state)]
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

    def get_controls_from_finite_memory_str_minigrid(self,
                                                     graph,
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

        _start_state = graph.get_initial_states()[0][0]
        _total_human_intervention = _start_state[0][1]
        _accepting_states = graph.get_accepting_states()
        _trap_states = graph.get_trap_states()

        _absorbing_states = _accepting_states + _trap_states
        _human_interventions: int = 0
        _visited_states = []
        _position_sequence = []

        curr_sys_node = _start_state
        next_env_node = str_dict[curr_sys_node]

        _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
        next_sys_node = self._epsilon_greedy_finite_choose_action(graph,
                                                                  next_env_node,
                                                                  str_dict,
                                                                  epsilon=epsilon,
                                                                  _absoring_states=_absorbing_states,
                                                                  human_can_intervene=_can_human_intervene)[0]

        (x, y) = next_sys_node[0][0][0][0]
        _human_interventions: int = _total_human_intervention - next_sys_node[0][0][0][1]

        next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)

        _visited_states.append(curr_sys_node)
        _visited_states.append(next_sys_node)

        _entered_absorbing_state = False

        while 1:
            _position_sequence.append(next_pos)

            curr_sys_node = next_sys_node
            next_env_node = str_dict[curr_sys_node]

            if curr_sys_node == next_env_node:
                x, y = next_sys_node[0][0][0][0]
                _human_interventions: int = _total_human_intervention - next_sys_node[0][0][0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(next_sys_node)
                break

            # update the next sys node only if you not transiting to an absorbing state
            if next_env_node not in _absorbing_states and\
                    graph.get_state_w_attribute(next_env_node, "player") == "adam":
                _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
                next_sys_node = self._epsilon_greedy_finite_choose_action(graph,
                                                                          next_env_node,
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
                x, y = next_sys_node[0][0][0][0]
                _human_interventions: int = _total_human_intervention - next_sys_node[0][0][0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(next_sys_node)

        if not _entered_absorbing_state:
            _position_sequence.append(next_pos)

        if debug:
            print([_n for _n in _position_sequence])

        return _position_sequence

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
        if _start_state[0] == 'Init':
            _next_states = [_next_n for _next_n in self.graph._graph.successors(_start_state)]
            _start_state = _next_states[0]
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
