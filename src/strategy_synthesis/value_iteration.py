import warnings
import sys
import math
import networkx as nx
import numpy as np
import copy
import operator
import random

from collections import defaultdict, deque
from networkx import DiGraph
from numpy import ndarray
from bidict import bidict
from typing import Optional, Union, Dict, List, Tuple

# import local packages
from ..graph import graph_factory
from ..graph import TwoPlayerGraph
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

        # if there is no trap state or nothing transits to the trap state then initialize the accepting state as the
        # target state
        if len(_trap_states) == 0:
            warnings.warn("Trap state not found: Initializing cooperative game with accepting state as the target"
                          " vertex")
            self._initialize_target_state_costs()

        for _trap_state in _trap_states:
            if len(list(self.org_graph._graph.predecessors(_trap_state))) == 1:
                warnings.warn("Trap state not found: Initializing cooperative game with accepting state as the target"
                              " vertex")
                self._initialize_target_state_costs()


        for _s in _trap_states:
            _node_int = self.node_int_map[_s]
            self.val_vector[_node_int][0] = 0

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
        self._initialize_target_state_costs()

    def _is_same(self, pre_val_vec, curr_val_vec):
        """
        A method to check is if the two value vectors are same or not
        :param pre_val_vec:
        :param curr_val_vec:
        :return:
        """

        return np.array_equal(pre_val_vec, curr_val_vec)

    def _add_trap_state_player(self):
        """
        A method to add a player to the trap state, if any
        :return:
        """

        _trap_states = self.org_graph.get_trap_states()

        for _n in _trap_states:
            self.org_graph.add_state_attribute(_n, "player", "adam")

    def online_reg_solver(self, cval: dict, debug: bool = False, plot: bool = False):
        """
        This is a online variant of computing regret. In this algorithm, as states converge to their state values we
        subtract the cooperative values associated with that state. We then lock the state value and strategy for that
        state to avoid/resetting the state value to itself.

        :param cval: A dict that maps a state in the game to its corresponding cooperative value
        :param debug:
        :param plot:
        :return:
        """

        # initially in the org val_vector the target node(s) will value 0
        _accp_state = self.org_graph.get_accepting_states()[0]
        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]

        self._add_trap_state_player()

        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=INT_MAX_VAL, dtype=np.int32)

        iter_var = 0
        _max_str_dict = {}
        _min_str_dict = {}
        _min_reach_str_dict = {}
        _max_reach_str_dict = {}

        converged_states = deque()

        while not self._is_same(_val_pre, _val_vector):
            if debug:
                if iter_var % 1000 == 0:
                    print(f"{iter_var} Iterations")
                    # print(f"Init state value: {self.val_vector[_init_int_node]}")

            _val_pre = copy.copy(_val_vector)
            iter_var += 1

            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state or _n in converged_states:
                    converged_states.append(_n)
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

            # check if any of the state have converged
            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state\
                        or _n in converged_states\
                        or self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                    continue

                # if a state has converged we then subtract that cooperative value from that state
                if _val_pre[_int_node] != INT_MAX_VAL and\
                        _val_vector[_int_node] == _val_pre[_int_node] and\
                        self._check_state_converged(_n, converged_states):
                    # _val_vector[_int_node] = _val_vector[_int_node] - cval[_n]
                    _val_vector[_int_node] = cval[_n] - _val_vector[_int_node]
                    converged_states.append(_n)

                    # note: for now check if any state value become negative or not because of this computation
                    if debug and _val_vector[_int_node] < 0:
                        print(f"state {_n} in iteration {iter_var} is assigned {_val_vector[_int_node]} after"
                              f"subtracting its cooperative value")

            self._val_vector = np.append(self.val_vector, _val_vector, axis=1)

        _str_dict = {**_max_str_dict, **_min_str_dict}

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: self.val_vector[i][iter_var]})

        if plot:
            self._add_state_costs_to_graph()
            self.add_str_flag(_str_dict)
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            self._sanity_check()
            # self.print_state_values()

        return _str_dict

    def _check_state_converged(self, state: Union[Tuple, str], converged_states: deque) -> bool:
        """
        A helper method called by the online_reg_solver function to check if a state has converged or not. We call a
        state converged when all its neighbouring states have also converged i.e all the neighbouring states in the
        converged_states queue. We return the value as True if the states have converged else we return False.
        :return: bool value
        """

        # if there are no neighbour then will automatically return true
        _converged = True

        for _n in self.org_graph._graph.neighbors(state):
            if _n in converged_states:
                _converged = True
            else:
                _converged = False
                break

        return _converged

    def solve(self, debug: bool = False, plot: bool = False):
        """
        A method that implements Algorithm 1 from the paper. The operation performed at each step can be represented by
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

        self._add_trap_state_player()

        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=INT_MAX_VAL, dtype=np.int32)

        iter_var = 0
        _max_str_dict = {}
        _min_str_dict = {}
        _min_reach_str_dict = {}
        _max_reach_str_dict = {}

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

        _str_dict = {**_max_str_dict, **_min_str_dict}

        if plot:
            self._add_state_costs_to_graph()
            self.add_str_flag(_str_dict)
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            self._sanity_check()
            # self.print_state_values()

        return _str_dict

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
            _val = (_node_int, self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0])
            if pre_vec[_node_int][0] == INT_MAX_VAL:
                _succ_vals.append((_node_int, INT_MAX_VAL))
            else:
                # _val = (_node_int, pre_vec[_node_int][0])
                _succ_vals.append(_val)

        # get org node int value
        if self.competitive:
            _next_node_int, _val = max(_succ_vals, key=operator.itemgetter(1))
        else:
            _next_node_int, _val = min(_succ_vals, key=operator.itemgetter(1))

        _curr_node_int = self.node_int_map[node]
        if INT_MIN_VAL <= _val <= INT_MAX_VAL:
            return _val, _next_node_int

        return pre_vec[_curr_node_int][0], _next_node_int

    def _add_state_costs_to_graph(self):
        """
        A helper method that computes the costs associated with each state to reach the accepting state and add it to
        the nodes.
        :return:
        """

        for _n in self.org_graph._graph.nodes():
            self.org_graph.add_state_attribute(_n, "ap", self.state_value_dict[_n])

    def add_str_flag(self, str_dict: Dict):
        """

        :param str_dict:
        :return:
        """
        self.org_graph.set_edge_attribute('strategy', False)

        for curr_node, next_node in str_dict.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    self.org_graph._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.org_graph._graph.edges[curr_node, next_node, 0]['strategy'] = True

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray,)\
            -> Tuple[Union[int, float], int]:
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
            if pre_vec[_node_int][0] == INT_MAX_VAL:
                _succ_vals.append((_node_int, INT_MAX_VAL))
            else:
                _succ_vals.append((_node_int, _val))

        _next_node_int, _val = min(_succ_vals, key=operator.itemgetter(1))

        if INT_MIN_VAL <= _val <= INT_MAX_VAL:
            return _val, _next_node_int

        _curr_node_int = self.node_int_map[node]
        _val = pre_vec[_curr_node_int][0]

        return _val, _next_node_int