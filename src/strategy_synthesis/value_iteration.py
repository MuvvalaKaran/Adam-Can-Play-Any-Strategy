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
from typing import Optional, Union, Dict, List

# import local packages
from src.graph import TwoPlayerGraph
# from src.strategy_synthesis import ReachabilitySolver
from .adversarial_game import ReachabilityGame as ReachabilitySolver

# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class ValueIteration:

    def __init__(self, game: TwoPlayerGraph, competitve: bool = False):
        self.org_graph: Optional[TwoPlayerGraph] = copy.deepcopy(game)
        self.competitive = competitve
        self._local_graph: Optional[DiGraph] = None
        self._val_vector: Optional[ndarray] = None
        self._node_int_map: Optional[bidict] = None
        self._num_of_nodes: int = 0
        self._W: int = 0
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

    def _convert_weights_to_positive_costs(self, plot: bool = False, debug: bool = False):
        """
        A helper method that converts the -ve weight that represent cost to positive edge weights for a given game.
        :return:
        """

        for _e in self.org_graph._graph.edges.data("weight"):
            _u = _e[0]
            _v = _e[1]

            _curr_weight = _e[2]
            if _curr_weight < 0:
                _new_weight: Union[int, float] = -1 * _curr_weight
            else:
                if debug:
                    print(f"Got a positive weight in the graph for edge {_u}------>{_v} with edge weight"
                          f" {_curr_weight}")
                _new_weight: Union[int, float] = _curr_weight

            self.org_graph._graph[_u][_v][0]["weight"] = _new_weight

        if plot:
            self.org_graph.plot_graph()

    def _initialize_target_state_costs(self):
        """
        A method that computes the set of target states, and assing the respective nodes a zero value in the the value
        vector
        :return:
        """
        _accp_state = self.org_graph.get_accepting_states()

        for _s in _accp_state:
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
        self._val_vector = np.full(shape=(self.num_of_nodes, 1), fill_value=140, dtype=np.int32)

        # if self.org_graph._graph_name != 'G_hat':
        self._convert_weights_to_positive_costs(plot=False, debug=False)
        self._add_accp_states_self_loop_zero_weight()
        self._initialize_target_state_costs()

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
        A method that manully adds an edge weight of zero to all the edges that transit to the absorbing states
        :return:
        """
        _accp_state = self.org_graph.get_accepting_states()
        _trap_state = self.org_graph.get_trap_states()

        for _s in _accp_state + _trap_state:
            for _pre_s in self.org_graph._graph.predecessors(_s):
                if _pre_s != _s:
                    self.org_graph._graph[_pre_s][_s][0]['weight'] = 0

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
        _trap_state = self.org_graph.get_trap_states()[0]
        # _trap_int_node = self.node_int_map[_trap_state]

        self.org_graph.add_state_attribute(_trap_state, "player", "eve")
        self._add_edges_to_abs_states_as_zero()
        # assign edges that transit to the absorbing states to be zero.
        # _edges = []
        # for _n in [_trap_state] + [_accp_state]:
        #     for _pre_n in self.org_graph._graph.predecessors(_n):
        #         if _pre_n != _n:
        #             self.org_graph._graph[_pre_n][_n][0]['weight'] = 0

        _val_vector = self.val_vector
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=140, dtype=np.int32)

        iter_var = 0

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
                    _val_vector[_int_node][0] = self._get_max_env_val(_n, _val_pre)

                elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                    _val_vector[_int_node][0] = self._get_min_sys_val(_n, _val_pre)

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

        if plot:
            self._add_state_costs_to_graph()
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.val_vector[_init_int_node][iter_var]}")
            # self.print_state_values()

    def _compute_convergence_idx(self) -> Dict[int, int]:
        """
        This method is used to determin when each state in the graph converged to their values. A state value is
        converged if x_{k+1} = x_k where k is the kth ieration and x is a state in the graph
        :return:
        """
        _convergence_dict: dict = {}
        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]
        _init_val = self.val_vector[_init_int_node][0]

        _num_of_states, _num_of_iter = self.val_vector.shape

        for _state in range(_num_of_states):
            # fails safe mechanism for states that retain their initial value of max positive integer. This means that
            # from this state you are bound to end up in the trap state. For such states, we add convergence value to be
            # the max value of iteration - kinda like saying never converged
            if self.val_vector[_state][0] == self.val_vector[_state][-1]:
                _convergence_dict.update({_state: _num_of_iter})
                continue

            for _itr in range(_num_of_iter - 1):
                if self.val_vector[_state][_itr] != _init_val:
                    if self.val_vector[_state][_itr + 1] == self.val_vector[_state][_itr]:
                        _convergence_dict.update({_state: _itr})
                        break

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
        _trap_state = self.org_graph.get_trap_states()[0]
        # _pre_trap_states = []
        _pre_sys_trap_states = set()
        for _pre_n in self.org_graph._graph.predecessors(_trap_state):
            # _pre_trap_states.append(_pre_n)
            # compute human states that have transition to this pre
            if _pre_n != _trap_state:
                for _pre_sys in self.org_graph._graph.predecessors(_pre_n):
                    _pre_sys_trap_states.add(_pre_sys)

        return _pre_sys_trap_states

    def compute_strategies(self, max_prefix_len: int = 3):

        _str_dict = {}
        if not isinstance(max_prefix_len, int) or max_prefix_len < 0:
            warnings.warn("The memory for sys player should be a semi-positive integer(>= 0)")
            sys.exit(-1)

        print("Computing Strategies for Eve and Adam")
        conv_dict: Dict[int, int] = self._compute_convergence_idx()
        _states_to_avoid = self.__compute_states_to_avoid()

        # compute the set of states that belong to the attractor region
        adv_solver = ReachabilitySolver(self.org_graph)
        adv_solver.reachability_solver()
        attr_region = adv_solver.sys_winning_region
        sys_str = adv_solver.sys_str

        for _n in self.org_graph._graph.nodes():
            _int_node = self.node_int_map[_n]

            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                strategy = self.get_str_for_env(_n)
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                # if _n in attr_region:
                #     strategy = sys_str[_n]
                # else:
                _conv_at = conv_dict.get(_int_node)
                strategy = self.get_str_for_sys(_n, max_prefix_len, _conv_at, _states_to_avoid)
            else:
                warnings.warn(f"State {_n} does not have a valid player associated wiht it.")
                continue
                # sys.exit(-1)

            _str_dict.update({_n: strategy})

        print("Done Computing Strategies for Eve and Adam")

        return _str_dict

    def get_str_for_env(self, node: Union[str, tuple]) -> Union[str, tuple]:
        """
        As MAX player or env player in our case has a memoryless strategy we return the next node
        :param node:
        :return:
        """
        _succ_vals = []
        for _next_n in self.org_graph._graph.successors(node):
            val = self.org_graph.get_edge_weight(node, _next_n) + self.state_value_dict[_next_n]
            _succ_vals.append((_next_n, val))

        return max(_succ_vals, key=operator.itemgetter(1))[0]

    def get_str_for_sys(self, node: Union[str, tuple], max_prefix_len: int, convg_at: int, states_to_avoid: set):
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
            _next_node = self._get_min_sys_node(node, _val_vector, states_to_avoid)
            _state_dict.update({prefix_len: _next_node})

        # for prefixes whose len > max_prefix_len we use the second cond
        _val_vector = self.val_vector[:, 0]
        _next_node = self._get_min_sys_node(node, _val_vector, states_to_avoid)
        _state_dict.update({max_prefix_len + 1: _next_node})

        return _state_dict

    def _get_min_sys_node(self,  node: Union[str, tuple], pre_vec: ndarray, states_to_avoid: set):
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
        _all_max_vals = True
        _all_states_lead_to_trap = True
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]

            if pre_vec[_node_int] != INT_MAX_VAL:
                _all_max_vals = False

            if _next_n not in states_to_avoid:
                _all_states_lead_to_trap = False

            _succ_vals.append((_next_n, val))

        # if all the values in the list are MAX_INT_VAL then pick an edge randomly. Also make sure that edge does not
        # lead to a state that only has an edge to the trap state
        if _all_max_vals and not _all_states_lead_to_trap:
            _node = random.choice(_succ_vals)[0]
            while _node in states_to_avoid:
                _node = random.choice(_succ_vals)[0]
            return _node

        return min(_succ_vals, key=operator.itemgetter(1))[0]

    def print_state_values(self):
        """
        A method to print the state value
        :return:
        """

        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            print(f"State {_s} Value {self.val_vector[i]}")

    def _get_max_env_val(self, node: Union[str, tuple], pre_vec: ndarray) -> Union[int, float]:
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
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0]
            _succ_vals.append(val)

        # get org node int value
        if self.competitive:
            val = max(_succ_vals)
        else:
            val = min(_succ_vals)

        _node_int = self.node_int_map[node]
        if INT_MIN_VAL <= val <= INT_MAX_VAL:
            return val

        return pre_vec[_node_int][0]

    def _add_state_costs_to_graph(self):
        """
        A helper method that computes the costs associated with each state to reach the accepting state and add it to
        the nodes.
        :return:
        """

        for _n in self.org_graph._graph.nodes():
            self.org_graph.add_state_attribute(_n, "ap", self.state_value_dict[_n])

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray) -> Union[int, float]:
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
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0]
            _succ_vals.append(val)

        _node_int = self.node_int_map[node]
        if INT_MIN_VAL <= min(_succ_vals) <= INT_MAX_VAL:
            return min(_succ_vals)

        return pre_vec[_node_int][0]