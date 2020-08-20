# Array Based FAS implementation
import warnings
import sys

import networkx as nx
import numpy as np

from numpy import ndarray
from typing import Optional, Union, Dict, List, Set
from networkx import DiGraph

# import local packages
from solvers import Solver
from factory.builder import Builder

INT_MIN_VAL = -100000


class ArrayFAS(Solver):
    """
    A implementation of the greedy FAS solver based on section 2.3 from [1].

    Identify all the sources and sinks in the graph(G) iteratively. If the vertex, say u, removed from G is a sink then
    it is prepended to sequence s2, otherwise, it is appended to a sequence s1.

    s =  s1s2 [Source Sinks nodes]. This sequence represents a linear arrangement of G for which the backward arcs make
    up the Feedback Arc Set (FAS). The pseudocode is given in Algo 1.

    Parameters
    ----------

    graph:         Digraph      The input graph(G). All nodes of this graph are assumed to positive integers.
                                There does not exists multiple edges any two nodes (DiGraph!). Preferably no self-loops.

    n:             int          Total number of nodes in G

    num_classes:   int          Total Number of vertex classes

    bins:          ndarray      A Flat array that holds the tail of bin i

    prev:          ndarray      A Flat array that holds the reference to the previous node of vertex i

    next:          ndarray      A Flat array that holds the reference to the next node of vertex i

    deltas:        ndarray      Delta class for the vertex. Used to determine if a node is present or not

    For mutators and accessors related material visit : https://www.python-course.eu/python3_properties.php
    """
    def __init__(self, graph: DiGraph):
        Solver.__init__(self, graph)

        self.__n: int = 0
        self.__num_classes: int = 0
        self.__bins:  Optional[ndarray] = None
        self.__prev: Optional[ndarray] = None
        self.__next: Optional[ndarray] = None
        self.__deltas: Optional[ndarray] = None
        self.__max_delta: int = INT_MIN_VAL
        self.__seq: List[int] = []
        self._initialize_solver()

    # @property
    # def graph(self):
    #     return self.__graph

    @property
    def n(self):
        return self.__n

    @property
    def num_classes(self):
        return self.__num_classes

    @property
    def bins(self):
        return self.__bins

    @property
    def prev(self):
        return self.__prev

    @property
    def next(self):
        return self.__next

    @property
    def deltas(self):
        return self.__deltas

    @property
    def max_delta(self):
        return self.__max_delta

    @property
    def seq(self):
        return self.__seq

    # @graph.setter
    # def graph(self, graph):
    #     if len(graph.nodes()) == 0:
    #         warnings.warn("Please make sure that the graph is not empty")
    #         sys.exit(-1)
    #
    #     if not isinstance(graph, DiGraph):
    #         warnings.warn("The Graph should be of type of DiGraph from networkx package")
    #         sys.exit(-1)
    #
    #     self.__graph = graph

    @n.setter
    def n(self, value: int):
        self.__n = value

    @max_delta.setter
    def max_delta(self, value: int):
        self.__max_delta = value

    @num_classes.setter
    def num_classes(self, value: int):
        self.__num_classes = value

    @deltas.setter
    def deltas(self, value: ndarray):
        self.__deltas = value

    @next.setter
    def next(self, value: ndarray):
        self.__next = value

    @prev.setter
    def prev(self, value: ndarray):
        self.__prev = value

    @bins.setter
    def bins(self, value: ndarray):
        self.__bins = value

    @seq.setter
    def seq(self, value: List):
        self.__seq = value

    def _initialize_solver(self):
        self.n = len(self.graph.nodes())
        self.num_classes = 2*self.n - 3
        self.deltas = np.empty(shape=(self.n, ), dtype=np.int32)
        self.next = np.full(shape=(self.n, ), fill_value=-1, dtype=np.int32)
        self.prev = np.full(shape=(self.n, ), fill_value=-1, dtype=np.int32)
        self.bins = np.full(shape=(self.num_classes, ), fill_value=-1, dtype=np.int32)

        self.__create_bins()

    def __create_bins(self):

        for _n in self.graph.nodes():

            _out_deg: int = self._get_out_degrees(_n)
            _in_deg: int = self._get_in_degrees(_n)

            if _out_deg == 0:
                self._add_to_bin(2 - self.n, _n)
                self.deltas[_n] = 2 - self.n
            elif _in_deg == 0 and _out_deg > 0:
                self._add_to_bin(self.n - 2, _n)
                self.deltas[_n] = self.n - 2
            else:
                class_d: int = _out_deg - _in_deg
                self._add_to_bin(class_d, _n)
                self.deltas[_n] = class_d

    def _get_out_degrees(self, node: int) -> int:

        _num_out_degree: int = 0
        for _n in self.graph.successors(node):
            if _n == node:
                continue
            _num_out_degree += 1

        return _num_out_degree

    def _get_in_degrees(self, node: int) -> int:

        _num_in_degree: int = 0
        for _n in self.graph.predecessors(node):
            if _n == node:
                continue
            _num_in_degree += 1

        return _num_in_degree

    def _add_to_bin(self, delta: int, node: int):
        """
        A method that the corresponding node to its appropriate class based on the delta value

        delta = _out_degree - _in_degree
        :param delta:
        :param node:
        :return:
        """

        # if the bin does'nt have any node
        if self.bins[delta - (2 - self.n)] == -1:
            self.bins[delta - (2 - self.n)] = node
            self.prev[node] = -1
        else:
            self.next[self.bins[delta - (2 - self.n)]] = node
            self.prev[node] = self.bins[delta - (2 - self.n)]
            self.bins[delta - (2 - self.n)] = node

        self.next[node] = -1

        if delta < self.n - 2 and self.max_delta < delta:
            self.max_delta = delta

    def _update_max_delta(self, delta: int):

        if delta == self.max_delta and self.bins[delta - (2 - self.n)] == -1:
            while self.bins[self.max_delta - (2 - self.n)] == -1:
                self.max_delta -= 1

                if self.max_delta == (2 - self.n):
                    break

    def _delete_node(self, node: int):
        """
        A method to delete a vertex from G
        :param node:
        :return:
        """
        self.deltas[node] = INT_MIN_VAL

        # delete node from G and update the Vertex class and bin of is neighbours in G
        self._update_node_class_bins(self.graph, node, True)
        self._update_node_class_bins(self.graph.reverse(), node, False)

        self.prev[node] = -1
        self.next[node] = -1

    def _update_node_class_bins(self, graph: DiGraph,  u_node: int, out: bool):

        for _v_node in graph.successors(u_node):
            if _v_node == u_node:
                continue

            if self.deltas[_v_node] > INT_MIN_VAL:
                _old_delta: int = self.deltas[_v_node]
                _new_delta: int = _old_delta

                if out:
                    _new_delta += 1
                else:
                    _new_delta -= 1

                self.deltas[_v_node] = _new_delta

                if self.bins[_old_delta - (2 - self.n)] == _v_node:
                    self.bins[_old_delta - (2 - self.n)] = self.prev[_v_node]

                if self.prev[_v_node] != -1:
                    self.next[self.prev[_v_node]] = self.next[_v_node]

                if self.next[_v_node] != -1:
                    self.prev[self.next[_v_node]] = self.prev[_v_node]

                self._add_to_bin(_new_delta, _v_node)
                self._update_max_delta(_old_delta)

    def _compute_seq(self):
        """
        A method to compute to s = s1s2 sequence - [Source state; Sink states] for G
        :return:
        """

        s1: List[int] = []
        s2: List[int] = []

        num_del = 0

        while num_del < self.n:
            while self.bins[0] != -1:
                _node = self.bins[0]
                self.bins[0] = self.prev[_node]

                if self.prev[_node] != -1:
                    self.next[self.prev[_node]] = -1

                self._delete_node(_node)
                num_del += 1
                s2.insert(0, _node)

            while self.bins[self.num_classes - 1] != -1:
                _node = self.bins[self.num_classes - 1]
                self.bins[self.num_classes - 1] = self.prev[_node]

                if self.prev[_node] != -1:
                    self.next[self.prev[_node]] = -1

                self._delete_node(_node)
                num_del += 1
                s1.append(_node)

            if num_del < self.n:
                _node = self.bins[self.max_delta - (2 - self.n)]
                self.bins[self.max_delta - (2 - self.n)] = self.prev[_node]

                if self.prev[_node] != -1:
                    self.next[self.prev[_node]] = -1

                self._update_max_delta(self.max_delta)
                self._delete_node(_node)

                num_del += 1
                s1.append(_node)

        self.seq = s1 + s2

    def solve(self, debug: bool = False):
        """
        A method to create the DAG from the computed vertex sequence s = s1s2
        :return:
        """
        if len(self.seq) == 0:
            self._compute_seq()

        v_array: List[int] = [-1 for i in range(self.n)]

        for ixs, _s in enumerate(self.seq):
            v_array[_s] = ixs

        fvs_set = set()
        fas_set = set()
        fas_count: int = 0

        for _n in self.graph.nodes():

            for _next_n in self.graph.successors(_n):

                if _next_n == _n:
                    continue

                if v_array[_n] > v_array[_next_n]:
                    fvs_set.add(_n)
                    fas_set.add((_n, _next_n))
                    fas_count += 1

        if debug:
            print(f"The size of fas is : {fas_count}")
            print(f"Fas is {[i for i in fas_set]}")
            print(f"Fvs is {[i for i in fvs_set]}")


class ArrayFASBuilder(Builder):
    """
    Implements the generic Solver builder for ArrayFas algorithm
    """
    def __init__(self):
        Builder.__init__(self)

    def __call__(self, graph: DiGraph):

        self._instance = ArrayFAS(graph)

        return self._instance






