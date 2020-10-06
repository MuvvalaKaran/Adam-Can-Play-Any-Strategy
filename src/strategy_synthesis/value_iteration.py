import warnings
import sys
import math
import networkx as nx
import numpy as np
import copy
import operator
import random

from collections import defaultdict
from networkx import DiGraph
from numpy import ndarray
from bidict import bidict
from typing import Optional, Union, Dict, List, Tuple

# import local packages
from src.graph import graph_factory
from src.graph import TwoPlayerGraph
# from src.strategy_synthesis import ReachabilitySolver
from .adversarial_game import ReachabilityGame as ReachabilitySolver

# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class ValueIteration:

    def __init__(self, game: TwoPlayerGraph, competitive: bool = False):
        self.org_graph: Optional[TwoPlayerGraph] = copy.deepcopy(game)
        self.competitive = competitive
        self._local_graph: Optional[DiGraph] = None
        self._val_vector: Optional[ndarray] = None
        self._node_int_map: Optional[bidict] = None
        self._num_of_nodes: int = 0
        self._MAX_POSSIBLE_W: int = 0
        self.W: int = 0
        self._state_value_dict: Optional[dict] = defaultdict(lambda: -1)
        self._initialize_val_vector()

    @property
    def org_graph(self):
        return self.__org_graph

    @property
    def val_vector(self):
        return self._val_vector

    @property
    def competitive(self):
        return self._competitive

    @property
    def num_of_nodes(self):
        return self._num_of_nodes

    @property
    def W(self):
        return self._W

    @property
    def local_graph(self):
        return self._local_graph

    @property
    def state_value_dict(self):
        return self._state_value_dict

    @property
    def node_int_map(self):
        return self._node_int_map

    @org_graph.setter
    def org_graph(self, org_graph):
        if len(org_graph._graph.nodes()) == 0:
            warnings.warn("Please make sure that the graph is not empty")
            sys.exit(-1)

        if not isinstance(org_graph, TwoPlayerGraph):
            warnings.warn("The Graph should be of type of TwoPlayerGraph")
            sys.exit(-1)

        self.__org_graph = org_graph

    @competitive.setter
    def competitive(self, value: bool):
        self._competitive = value

    @W.setter
    def W(self, value: int):
        self._W = value

    def _convert_weights_to_positive_costs(self, plot: bool = False, debug: bool = False):
        """
        A helper method that converts the -ve weight that represent cost to positive edge weights for a given game.
        :return:
        """

        for _e in self.org_graph._graph.edges.data("weight"):
            _u = _e[0]
            _v = _e[1]

            _curr_weight = _e[2]
            # if _curr_weight < 0:
            _new_weight: Union[int, float] = -1 * _curr_weight
            # else:
            #     if debug:
            #         print(f"Got a positive weight in the graph for edge {_u}------>{_v} with edge weight"
            #               f" {_curr_weight}")
            #     _new_weight: Union[int, float] = _curr_weight

            self.org_graph._graph[_u][_v][0]["weight"] = _new_weight

        if plot:
            self.org_graph.plot_graph()

    def _initialize_target_state_costs(self):
        """
        A method that computes the set of target states, and assigns the respective nodes a zero value in the the value
        vector
        :return:
        """
        _accp_state = self.org_graph.get_accepting_states()

        for _s in _accp_state:
            _node_int = self.node_int_map[_s]
            self.val_vector[_node_int][0] = 0

    def _initialize_trap_state_costs(self):
        """
        A method that computes the set of trap states, and assigns the respective node a zero value in the value vector.
        The weights transiting to trap state however are given a big finite value in this case. This is done to make
        sure the values associated with states from where you cannot reach the accepting state(s) (The trap region in an
        adversarial game) have a finite (but big) values rather than all having the states in the trap region having inf
        values
        :return:
        """

        _trap_states = self.org_graph.get_trap_states()

        for _s in _trap_states:
            _node_int = self.node_int_map[_s]
            self.val_vector[_node_int][0] = 0

    def _add_accp_states_self_loop_zero_weight(self):
        """
        For g_hat construction, the accpeting's self loop has finite value. This is because the org weight is 0.

        0 + b = some finite value and due to this self-loop weight, the algorithm explodes. In this method we manually
        write the self-loop weight to be zero
        :return:
        """
        _accp_states = self.org_graph.get_accepting_states()

        for _s in _accp_states:
            self.org_graph._graph[_s][_s][0]['weight'] = 0

    def _initialize_val_vector(self):
        """
        A method to initialize parameters such the
         1. Number of nodes in a given graph,
         2. The max absolute weight,
         3. The internal node mapping dict, and
         4. Initializing the value vector and assigning all the target state(s) 0 init value
        :return:
        """
        self._num_of_nodes = len(list(self.org_graph._graph.nodes))
        self._W = abs(self.org_graph.get_max_weight())
        self._node_int_map = bidict({state: index for index, state in enumerate(self.org_graph._graph.nodes)})
        self._val_vector = np.full(shape=(self.num_of_nodes, 1), fill_value=INT_MAX_VAL, dtype=np.int32)

        # if self.org_graph._graph_name != 'G_hat': `
        # if not self.competitive:
        self._convert_weights_to_positive_costs(plot=False, debug=False)
        # self._add_accp_states_self_loop_zero_weight()
        self._initialize_target_state_costs()
        # if self.competitive:
        #     self._initialize_trap_state_costs()

    def convert_graph_to_mcr_graph(self):
        """

        The input graph to Value Iteration(VI) algorithm internally maps all the nodes to an integer value.
        This is done to speed up the computation of the VI algorithm
        :return:
        """

        _graph = nx.DiGraph(name="local_scc_solver_graph")
        _graph.add_nodes_from(self._node_int_map.values())

        # add edges of the local graph
        for _e in self.org_graph._graph.edges():
            _u = self._node_int_map[_e[0]]
            _v = self._node_int_map[_e[1]]

            if _graph.has_edge(_u, _v):
                warnings.warn(f"The Graph has multiple edges from {_e[0]} to {_e[1]}."
                              f" This is not valid for the FAS solver graph as it is a Directed Graph -"
                              f" no multiple edges from u to v.")
                sys.exit(-1)

            _graph.add_edge(_u, _v)

        self._local_graph = _graph

    def _is_same(self, pre_val_vec, curr_val_vec):
        """
        A method to check is if the two value vectors are same or not
        :param pre_val_vec:
        :param curr_val_vec:
        :return:
        """

        return np.array_equal(pre_val_vec, curr_val_vec)

    def _add_edges_to_abs_states_as_zero(self):
        """
        A method that manually adds an edge weight of zero to all the edges that transit to the absorbing states from
        the sys states.
        :return:
        """

        _accp_state = self.org_graph.get_accepting_states()
        _trap_state = self.org_graph.get_trap_states()

        for _s in _accp_state + _trap_state:
            for _pre_s in self.org_graph._graph.predecessors(_s):
                if _pre_s != _s:
                    self.org_graph._graph[_pre_s][_s][0]['weight'] = 0

    def _add_edges_to_trap_states_as_max(self):
        """
        A method that manually adds a finite edge weight  to all the edges that transit to the trap states. As per
        the paper Section 3. There exists optimal strategies for both players whose outcome is a non-looping path.

        So considering you visit each state atleast and atmost once, the max cumulative value of play could:

        |V - 1| * W : "-1" because you remove the trap state and the target/accepting state itself and W is the max abs
        value on the graph.

        Note we assume that all the weights in the graph are positive.
        :return:
        """
        # self._MAX_POSSIBLE_W = (self.num_of_nodes - 2) * self.W

        # _accp_state = self.org_graph.get_accepting_states()
        _trap_state = self.org_graph.get_trap_states()

        for _s in _trap_state:
            for _pre_s in self.org_graph._graph.predecessors(_s):
                if _pre_s != _s:
                    self.org_graph._graph[_pre_s][_s][0]['weight'] = self._MAX_POSSIBLE_W

    def _add_trap_state_player(self):
        """
        A method to add a player to the trap state, if any
        :return:
        """

        _trap_states = self.org_graph.get_trap_states()

        for _n in _trap_states:
            self.org_graph.add_state_attribute(_n, "player", "adam")

    def solve(self, debug: bool = False, plot: bool = False):
        """
        A method tHat implements Algorithm 1 from the paper. The operation performed at each step can be represented by
        an operator say F.  F here is the _get_max_env_val() and _get_min_sys_val() methods. F is a monotonic operator
        and is monotonically decreasing - meaning the function should not increase (it must not increase)! and converges
        to the greatest fixed point of F.

        But as all the weight are positive in our case, the weight monotonically increase and converge to the greatest
        fixed point.

        The Val of the game is infact the Greatest Fixed Point.  The upper bound on the # of iterations to converge is
        (2|V| -1)W|V| + |V|.
        :param debug:
        :param plot:
        :return:
        """
        # initially in the org val_vector the target node(s) will value 0
        _accp_state = self.org_graph.get_accepting_states()[0]
        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]
        # if self.competitive:
        #     self.val_vector[_init_int_node][0] = 0
        # _trap_state = self.org_graph.get_trap_states()[0]
        # _trap_int_node = self.node_int_map[_trap_state]

        # self.org_graph.add_state_attribute(_trap_state, "player", "eve")
        self._add_trap_state_player()
        # self._add_edges_to_abs_states_as_zero()
        # if self.competitive:
        #     self._add_edges_to_trap_states_as_max()

        # assign edges that transit to the absorbing states to be zero.
        # _edges = []
        # for _n in [_trap_state] + [_accp_state]:
        #     for _pre_n in self.org_graph._graph.predecessors(_n):
        #         if _pre_n != _n:
        #             self.org_graph._graph[_pre_n][_n][0]['weight'] = 0

        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=INT_MAX_VAL, dtype=np.int32)

        iter_var = 0
        _max_str_dict = {}
        _min_str_dict = {}
        _min_reach_str_dict = {}

        while not self._is_same(_val_pre, _val_vector):
            if debug:
                if iter_var % 1000 == 0:
                    print(f"{iter_var} Iterations")
                    # print(f"Init state value: {self.val_vector[_init_int_node]}")
            _val_pre = copy.copy(_val_vector)
            iter_var += 1

            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state:
                    continue

                if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                    _val_vector[_int_node][0], _next_max_node = self._get_max_env_val(_n, _val_pre)
                    _max_str_dict[_n] = self.node_int_map.inverse[_next_max_node]

                elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                    _val_vector[_int_node][0], _next_min_node = self._get_min_sys_val(_n, _val_pre)
                    if _val_vector[_int_node] != _val_pre[_int_node]:
                        _min_str_dict[_n] = self.node_int_map.inverse[_next_min_node]

                        if _val_pre[_int_node] == INT_MAX_VAL:
                            _min_reach_str_dict[_n] = self.node_int_map.inverse[_next_min_node]

            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state:
                    continue

                if _val_vector[_int_node][0] < -1 * (self.num_of_nodes - 1) * self.W:
                    _val_vector[_int_node][0] = INT_MIN_VAL

            self._val_vector = np.append(self.val_vector, _val_vector, axis=1)

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: self.val_vector[i][iter_var]})

        # for v0 we have to specially make an exception if v1 has value higher than 0 then go down else take self-loop
        if self.competitive and self.org_graph._graph_name == "G_hat":
            _v1_node_int = self.node_int_map["v1"]
            if self.val_vector[_v1_node_int][0] >= 0:
                _max_str_dict["v0"] = "v1"
                self.state_value_dict["v0"] = self.state_value_dict["v1"]
            else:
                _max_str_dict["v0"] = "v0"

        if plot:
            self._add_state_costs_to_graph()
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            self._sanity_check()
            # self.print_state_values()

        return {**_max_str_dict, **_min_str_dict}

    def new_get_str_for_env(self):
        """
        A method to return the next
        :return:
        """

    def _sanity_check(self):
        """
        A nice charateristic of the algorithm is that the vertices from which you cannot reach the target set (assuming
        human to be purely adversarial) is exactly the region W2(the trap region) from the adversarial game solver. So
        this method checks if this property holds after the state values have computed
        :return:
        """
        adv_solver = ReachabilitySolver(self.org_graph)
        adv_solver.reachability_solver()
        _trap_region = set(adv_solver.env_winning_region)

        _num_of_states, _ = self.val_vector.shape

        _cumu_trap_region = set()
        for _i in range(_num_of_states):
            _node = self.node_int_map.inverse[_i]
            # if this state has INT_MAX_VAL as it final value then check it belong to the trap region
            if self.val_vector[_i][-1] == INT_MAX_VAL:
                _cumu_trap_region.add(_node)

        if _cumu_trap_region == _trap_region:
            print("The two sets are equal")

        # we know that trap region is correct so
        _missing_nodes = _trap_region.difference(_cumu_trap_region)

    def _compute_convergence_idx(self) -> Dict[int, int]:
        """
        This method is used to determine when each state in the graph converged to their values. A state value is
        converged if x_{k} != x_{k-1} where k is the kth iteration and x is a state in the graph
        :return:
        """
        _convergence_dict: dict = {}
        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]
        _init_val = self.val_vector[_init_int_node][0]

        _num_of_states, _num_of_iter = self.val_vector.shape

        # for _state in range(_num_of_states):
        #     # fails safe mechanism for states that retain their initial value of max positive integer. This means that
        #     # from this state you are bound to end up in the trap state. For such states, we add convergence value to be
        #     # the max value of iteration - kinda like saying never converged
        #     if self.val_vector[_state][0] == self.val_vector[_state][-1]:
        #         _convergence_dict.update({_state: _num_of_iter})
        #         continue
        #
        #     for _itr in range(_num_of_iter - 1):
        #         if self.val_vector[_state][_itr] != _init_val:
        #             if self.val_vector[_state][_itr + 1] == self.val_vector[_state][_itr]:
        #                 _convergence_dict.update({_state: _itr})
        #                 break

        for _state in range(_num_of_states):
            _converge_at_first_iter = True
            for _itr in range(_num_of_iter - 1, 0, -1):
                if self.val_vector[_state][_itr] != self.val_vector[_state][_itr - 1]:
                    _converge_at_first_iter = False
                    _convergence_dict.update({_state: _itr})
                    break
            if _converge_at_first_iter:
                _convergence_dict.update({_state: 0})

        return _convergence_dict

    def __compute_states_to_avoid(self) -> set:
        """
        A helper method that compute the set of states that the sys node should not voluntarily end up in.

        When all the nodes have the same val, the system currently picks a node voluntarily. In this random we would
        like to avoid picking state that will lead to the trap state. We compute the pre of trap states. To avoid
        visiting the pre, we need to avoid the human states that may lead to pre.

        e.g trap_state <---- sys_node <----- human_node <----sys_node

        This method compute the human_nodes in this sample series of nodes.
        :return:
        """

        # compute the list of states have transition to the trap state
        _trap_state = self.org_graph.get_trap_states()
        # _pre_trap_states = []
        _pre_sys_trap_states = set()
        for _s in _trap_state:
            for _pre_n in self.org_graph._graph.predecessors(_s):
                # _pre_trap_states.append(_pre_n)
                # compute human states that have transition to this pre
                if _pre_n != _s:
                    for _pre_sys in self.org_graph._graph.predecessors(_pre_n):
                        _pre_sys_trap_states.add(_pre_sys)

        return _pre_sys_trap_states

    def _build_trap_region_graph(self, trap_region: set, winning_region: set):
        """
        A method that build the trap region withtout the trap state and the sys state that leads to the trap state.

        Also we add a sink/target state for every edge that leads to the winning region with a self loop zero.

        Then we call the solve method to compute the values and compute the str on this graph. Note: all the nodes will
        have finite value and thus a strategy associated with it.
        :return:
        """
        _init_state = self.org_graph.get_initial_states()[0][0]
        two_player_graph = graph_factory.get("TwoPlayerGraph",
                                             graph_name="trap_region_graph",
                                             config_yaml="config/trap_region_graph",
                                             save_flag=True,
                                             pre_built=False,
                                             from_file=False,
                                             plot=False)

        def _check_if_human_state_is_equ_to_sys_state(human_state: tuple, env_state: tuple):
            """
            A inline metho that true if h(2,1)(3,1) and env_state is (3,1),".. " else False.
            This is used is trimming node in the trap region
            :param human_state:
            :param env_state:
            :return:
            """

            if isinstance(human_state, str) or self.org_graph.get_state_w_attribute(human_state, "player") != "adam":
                return False

            if isinstance(human_state, tuple):
                _human_str = human_state[0]
                _env_str = str(env_state[0])
                if not isinstance(_human_str, str):
                    _human_str = _human_str[0]
                    _env_str = str(env_state[0][0])
                    if not isinstance(_human_str, str):
                        _human_str = _human_str[0]
                        _env_str = str(env_state[0][0][0])

                if _human_str[0] == 'h':
                    if f"({_human_str.split('(')[2].split(')')[0]})" == _env_str:
                        return True

            return False


        # exclude the trap state the pre sys states that lead to the trap state
        # compute the list of states have transition to the trap state
        _states_to_be_pruned = set()
        _states_to_be_altered = set()
        _states_to_be_weighed_more = set()

        # compute the list of states have transition to the trap state
        _trap_state = self.org_graph.get_trap_states()
        # _pre_sys_trap_states = set()
        for _s in _trap_state:
            for _pre_n in self.org_graph._graph.predecessors(_s):
                # compute human states that have transition to this pre
                _states_to_be_pruned.add(_pre_n)
                for _pre_env in self.org_graph._graph.predecessors(_pre_n):
                    if _pre_env in trap_region:
                        # also prune human node that org transit to the states in the prune list
                        # e,g h(3,2)(3,1) where 3,1 is trap state
                        if _check_if_human_state_is_equ_to_sys_state(_pre_env, _pre_n):
                            _states_to_be_weighed_more.add(_pre_env)
                            _states_to_be_pruned.add(_pre_env)

                        _states_to_be_altered.add(_pre_env)

        # add a high weight from the sys nodes to the human nodes in_states_to_be_altered
        # as these states go very close to the trap state
        _states_to_be_altered = _states_to_be_altered - _states_to_be_pruned
        _sys_node_edges_to_be_altered = set()
        for _n in _states_to_be_altered:
            for _pre_sys in self.org_graph._graph.predecessors(_n):
                _sys_node_edges_to_be_altered.add((_pre_sys, _n))

        trap_region = trap_region - _states_to_be_pruned

        for _n in trap_region:
            _player = self.org_graph.get_state_w_attribute(_n, "player")
            two_player_graph.add_state(_n, player=_player)

        _accp_states = self.org_graph.get_accepting_states()
        for _accp_n in _accp_states:
            two_player_graph.add_state(_accp_n, player="eve")

        for _n in two_player_graph._graph.nodes():
            for _next_n in self.org_graph._graph.successors(_n):
                if _next_n in trap_region:
                    if (_n, _next_n) in _sys_node_edges_to_be_altered:
                        _weight = 100

                    # if(_n, _next_n) in _states_to_be_weighed_more:
                    #     _weight = 200

                    else:
                        _weight = self.org_graph.get_edge_weight(_n, _next_n)
                    two_player_graph.add_edge(_n, _next_n, weight=_weight)

        # for all the edges that transit to winning region add a sink/target state with a self-loop of zero there
        trans_edges = self.__get_edges_from_trap_to_winning_region(trap_region, winning_region)

        for _e in trans_edges:
            _u = _e[0]
            for _accp_n in _accp_states:
                two_player_graph.add_edge(_u, _accp_n, weight=0)
                two_player_graph._graph.add_edge(_accp_n, _accp_n, weight=0)

        for _accp_n in _accp_states:
            two_player_graph.add_accepting_state(_accp_n)

        two_player_graph.add_initial_state(_init_state)

        return two_player_graph, trap_region

    def _check_graph_is_total(self, graph: TwoPlayerGraph):
        """
        A method to check if the graph is total or not. If not then remove the node that don't have any out going edges.
        We again look at the grap and check for total-ness. We repeat the process until we reach a fix point
        :param graph:
        :return:
        """

        _converged = True
        _nodes_to_be_pruned = set()
        for _n in graph._graph.nodes():
            if len(list(graph._graph.successors(_n))) == 0:
                _converged = False
                _nodes_to_be_pruned.add(_n)

        graph._graph.remove_nodes_from(_nodes_to_be_pruned)

        if not _converged:
            self._check_graph_is_total(graph)

    def _get_trap_sys_nodes_str(self, trap_region: set, winning_region: set):
        """
        Given the trap region, compute the value of the game, and according compute the strategies
        :return:
        """

        _trap_graph, trap_region = self._build_trap_region_graph(trap_region, winning_region)
        self._check_graph_is_total(_trap_graph)
        solver = ValueIteration(_trap_graph, competitve=True)
        solver.solve()
        # str_dict = solver.state_value_dict
        str_dict = solver.compute_strategies()

        # we trim out all the str that have k >= 1 as we only want memoryless strategy
        for _s, val in str_dict.items():
            if isinstance(val, dict):
                str_dict[_s] = val[0]

        return str_dict, trap_region

    def __get_edges_from_trap_to_winning_region(self, trap_region: set, winning_region: set) -> set:
        """
        A method that returns the set of edges that transit from trap to winning region in a given game
        :param trap_region:
        :param winning_region:
        :return:
        """

        # loop over every node in the trap region and check if any one of its edges transit to the winning region
        _trans_edges = set()

        for _n in trap_region:
            for _edge in self.org_graph._graph.out_edges(_n):
                _v = _edge[1]
                if _v in winning_region:
                    _trans_edges.add(_edge)

        return _trans_edges

    def compute_strategies(self, max_prefix_len: int = 3):

        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]

        _str_dict = {}
        if not isinstance(max_prefix_len, int) or max_prefix_len < 0:
            warnings.warn("The memory for sys player should be a semi-positive integer(>= 0)")
            sys.exit(-1)

        print("Computing Strategies for Eve and Adam")
        conv_dict: Dict[int, int] = self._compute_convergence_idx()
        # _states_to_avoid = self.__compute_states_to_avoid()
        #
        # compute the sys nodes that lead to trap state for the env player. Refer the note in the method
        _trap_states = [i for i in self.org_graph.get_trap_states()]
        _sys_trap_states = set()
        for _t in _trap_states:
            for _pre_sys in self.org_graph._graph.predecessors(_t):
                _sys_trap_states.add(_pre_sys)

        # compute the set of states that belong to the attractor region
        adv_solver = ReachabilitySolver(self.org_graph)
        adv_solver.reachability_solver()
        sys_str = adv_solver.sys_str
        # trap_region = ()
        #
        # if self.org_graph._graph_name != "trap_region_graph" and self.val_vector[_init_int_node, -1] == INT_MAX_VAL:
        #     attr_region = adv_solver.sys_winning_region
        #     trap_region = set(adv_solver.env_winning_region)
        #     trap_str, trap_region = self._get_trap_sys_nodes_str(trap_region, attr_region)

        for _n in self.org_graph._graph.nodes():
            _int_node = self.node_int_map[_n]

            # if an node has only one edge then add that edge to the str
            _succ_n = [i for i in self.org_graph._graph.successors(_n)]
            if len(_succ_n) == 1:
                strategy = _succ_n[0]
                _str_dict.update({_n: strategy})
                continue

            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                strategy = self.get_str_for_env(_n, _sys_trap_states)
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                # if _n in trap_region:
                #     strategy = trap_str[_n]
                # else:
                _conv_at = conv_dict.get(_int_node)
                strategy = self.alt_str_for_sys(_n)
                # strategy = self.get_str_for_sys(_n, max_prefix_len, _conv_at, sys_str)
            else:
                warnings.warn(f"State {_n} does not have a valid player associated wiht it.")
                continue
                # sys.exit(-1)

            _str_dict.update({_n: strategy})

        print("Done Computing Strategies for Eve and Adam")

        return _str_dict

    def get_str_for_env(self, node: Union[str, tuple], sys_trap_states: set) -> Union[str, tuple]:
        """
        As MAX player or env player in our case has a memoryless strategy we return the next node
        :param node:
        :return:
        """
        # note while env player is naturally attracted to a state that leads to trap state, due to finiteness of the
        #  payoff, a adjacent cell from where the system cannot escape seems indifferent to the player - since they both
        #  have the max value INT_MAX_VAL. So I manuallay add a flag to check for sys state ------> trap state and choose
        #  them over any other state whenever possible
        _succ_vals = []
        for _next_n in self.org_graph._graph.successors(node):
            # val = self.org_graph.get_edge_weight(node, _next_n) + self.state_value_dict[_next_n]
            val = self.state_value_dict[_next_n]
            _succ_vals.append((_next_n, val))

        for _n, _ in _succ_vals:
            if _n in sys_trap_states:
                return _n

        if self.competitive:
            return max(_succ_vals, key=operator.itemgetter(1))[0]
        else:
            return min(_succ_vals, key=operator.itemgetter(1))[0]

    def alt_str_for_sys(self, node: Union[str, tuple]):
        """
        A memoryless strategy that only looks at the state values after they have converged
        :param node:
        :return:
        """
        _val_vector = self.val_vector[:, -1]
        _next_node = self._get_min_sys_node(node, _val_vector)

        return _next_node

    def get_str_for_sys(self, node: Union[str, tuple], max_prefix_len: int, convg_at: int, sys_str: dict):
        """
        A method that compute a finite memory strategy for the sys player. The strategy only depends on the length of
        the prefix and not the states we visited before. As such this method return a dict for each state as follows:

        _node : |prefix| = 1
                |prefix| = 2
                |prefix| = 3
                 :    :    :
                 :    :    :
                |prefix| = max_prefix_len

        The strategy for a prefix whose len is < k is :

            argmin {v' \in E(v)} (w(v, v') + x_{k - prefix_len - 1}(v'))

        for prefix whose len is >= k is :

            argmin {v' \in E(v)} (w(v, v') + x_0{v'})


        :param node:
        :param prefix_len:
        :return:
        """

        _state_dict: dict = {}
        for prefix_len in range(max_prefix_len):
            _val_vector = self.val_vector[:, convg_at - prefix_len - 1]
            _next_node = self._get_min_sys_node(node, _val_vector)
            _state_dict.update({prefix_len: _next_node})

        # for prefixes whose len > max_prefix_len we use the second cond
        # _val_vector = self.val_vector[:, 0]
        # _next_node = self._get_min_sys_node(node, _val_vector, set())
        _next_node = sys_str[node]
        _state_dict.update({max_prefix_len: _next_node})

        return _state_dict

    def _get_min_sys_node(self,  node: Union[str, tuple], pre_vec: ndarray):
        """
        A method that compute the next vertex which minimizes the formula w(v, v') + val_vect(v').

        NOTE: If all successor vertex have INT_MAX_VAL then we randomly select a node. This could be replaced with
         maybe a smarter technique where you find the shortest route to the node with a finite value i.e from where you
         could reach the accepting state.
        :param node:
        :param pre_vec:
        :param states_to_avoid:
        :return:
        """
        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            _succ_vals.append((_next_n, val))

        return min(_succ_vals, key=operator.itemgetter(1))[0]

    def print_state_values(self):
        """
        A method to print the state value
        :return:
        """

        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            print(f"State {_s} Value {self.val_vector[i]}")

    def _get_max_env_val(self, node: Union[str, tuple], pre_vec: ndarray) -> Tuple[Union[int, float], int]:
        """
        A method that returns the max value for the current node that belongs to the env.
        :param node: The current node in
        :param node_int: The mapped value of this node
        :param pre_vec: The previous value vector
        :return:
        """

        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            # val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0]
            _val = (_node_int, pre_vec[_node_int][0])
            _succ_vals.append(_val)

        # get org node int value
        if self.competitive:
            _next_node_int, _val = max(_succ_vals, key=operator.itemgetter(1))
        else:
            _next_node_int, _val = min(_succ_vals, key=operator.itemgetter(1))

        _curr_node_int = self.node_int_map[node]
        if INT_MIN_VAL <= _val <= INT_MAX_VAL:
            return _val, _next_node_int

        return pre_vec[_curr_node_int][0], _curr_node_int

    def _add_state_costs_to_graph(self):
        """
        A helper method that computes the costs associated with each state to reach the accepting state and add it to
        the nodes.
        :return:
        """

        for _n in self.org_graph._graph.nodes():
            self.org_graph.add_state_attribute(_n, "ap", self.state_value_dict[_n])

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray) -> Tuple[Union[int, float], int]:
        """
        A method that returns the min value of the current node that belongs to the sys
        :param node:
        :param node_int:
        :param pre_vec:
        :return:
        """

        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0]
            _succ_vals.append((_node_int, _val))

        _next_node_int, _val = min(_succ_vals, key=operator.itemgetter(1))

        _curr_node_int = self.node_int_map[node]
        if INT_MIN_VAL <= _val <= INT_MAX_VAL:
            return _val, _next_node_int

        return pre_vec[_curr_node_int][0], _curr_node_int