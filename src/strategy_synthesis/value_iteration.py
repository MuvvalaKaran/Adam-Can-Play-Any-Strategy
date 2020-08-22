import warnings
import sys
import math
import networkx as nx
import numpy as np
import copy
import operator

from collections import defaultdict
from networkx import DiGraph
from numpy import ndarray
from bidict import bidict
from typing import Optional, Union, Dict, List

# import local packages
from src.graph import TwoPlayerGraph

# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class ValueIteration:

    def __init__(self, game: TwoPlayerGraph):
        self.org_graph: Optional[TwoPlayerGraph] = copy.deepcopy(game)
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

    def _convert_weights_to_positive_costs(self, plot: bool = False):
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
                _new_weight: Union[int, float] = _curr_weight

            self.org_graph._graph[_u][_v][0]["weight"] = _new_weight

        if plot:
            self.org_graph.plot_graph()

    def _initialize_val_vector(self):


        _accp_state = self.org_graph.get_accepting_states()[0]

        self._num_of_nodes = len(list(self.org_graph._graph.nodes))
        self._W = abs(self.org_graph.get_max_weight())
        self._node_int_map = bidict({state: index for index, state in enumerate(self.org_graph._graph.nodes)})
        self._val_vector = np.full(shape=(self.num_of_nodes, ), fill_value=INT_MAX_VAL, dtype=np.int32)

        _accp_int_val = self.node_int_map[_accp_state]

        self.val_vector[_accp_int_val] = 0
        self._convert_weights_to_positive_costs(plot=False)

    def convert_graph_to_fas_graph(self):
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

    def solve(self, debug: bool = False, plot: bool = False):
        # initially in the org val_vector the target node will value 0
        _accp_state = self.org_graph.get_accepting_states()[0]
        _init_node = self.org_graph.get_initial_states()[0][0]
        _init_int_node = self.node_int_map[_init_node]
        _trap_state = self.org_graph.get_trap_states()[0]

        self.org_graph.add_state_attribute(_trap_state, "player", "eve")
        val_pre = np.full(shape=(self.num_of_nodes, ), fill_value=INT_MAX_VAL, dtype=np.int32)

        iter_var = 0

        while not self._is_same(val_pre, self.val_vector):

            if debug:
                if iter_var % 1000 == 0:
                    print(f"{iter_var} Iterations")
                    # print(f"Init state value: {self.val_vector[_init_int_node]}")
            iter_var += 1
            val_pre = copy.copy(self.val_vector)

            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state:
                    continue

                if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                    self.val_vector[_int_node] = self._get_max_env_val(_n, val_pre)

                elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                    self.val_vector[_int_node] = self._get_min_sys_val(_n, val_pre)

            for _n in self.org_graph._graph.nodes():
                _int_node = self.node_int_map[_n]

                if _n == _accp_state:
                    continue

                if self.val_vector[_int_node] < -1 * (self.num_of_nodes - 1) * self.W:
                    self.val_vector[_int_node] = INT_MIN_VAL

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: self.val_vector[i]})

        if plot:
            self._add_state_costs_to_graph()
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.val_vector[_init_int_node]}")
            # self.print_state_values()

    def compute_strategies(self):
        for _n in self.org_graph._graph.nodes():

            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                self.get_str_for_env(_n, 0)
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                pass
            else:
                warnings.warn(f"State {_n} does not have a valid player associated wiht it.")
                sys.exit(-1)

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

        return max(_succ_vals, key=operator.itemgetter(1))

    def get_str_for_sys(self, node: Union[str, tuple], prefix_len: int):
        pass

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
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            _succ_vals.append(val)

        # get org node int value
        _node_int = self.node_int_map[node]
        if INT_MIN_VAL <= max(_succ_vals) <= INT_MAX_VAL:
            return max(_succ_vals)

        return pre_vec[_node_int]

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
            val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            _succ_vals.append(val)

        _node_int = self.node_int_map[node]
        if INT_MIN_VAL <= min(_succ_vals) <= INT_MAX_VAL:
            return min(_succ_vals)

        return pre_vec[_node_int]