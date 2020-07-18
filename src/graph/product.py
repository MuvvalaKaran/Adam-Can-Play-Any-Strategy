from .two_player_graph import TwoPlayerGraph
from .trans_sys import FiniteTransSys
from .dfa import DFAGraph
from src.factory.builder import Builder

from typing import List, Tuple, Dict
from collections import deque, defaultdict
import networkx as nx

class ProductAutomaton(TwoPlayerGraph):

    def __init__(self, trans_sys_graph: FiniteTransSys, automaton: DFAGraph,
               graph_name: str, config_name, save_flag: bool = False):
        self._trans_sys = trans_sys_graph
        self._auto_graph = automaton
        self._graph_name = graph_name
        self._config_yaml = config_name
        self._save_flag = save_flag

    def construct_graph(self, absorbing: bool = False):
        super().construct_graph()
        max_w: str = self._trans_sys.get_max_weight()
        for _u_ts_node in self._trans_sys._graph.nodes():
            for _u_a_node in self._auto_graph._graph.nodes():
                if not absorbing:
                    _u_prod_node = self.composition(_u_ts_node, _u_a_node)
                else:
                    if _u_a_node in self._auto_graph.get_absorbing_states():
                        _u_prod_node = self.add_prod_state(_u_a_node, _u_a_node)
                    else:
                        _u_prod_node = self.composition(_u_ts_node, _u_a_node)

                for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):
                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):
                        if not absorbing:
                            _v_prod_node = self.composition(_v_ts_node, _v_a_node)
                        else:
                            if _v_a_node in self._auto_graph.get_absorbing_states():
                                _v_prod_node = self.add_prod_state(_v_a_node, _v_a_node)
                            else:
                                _v_prod_node = self.composition(_v_ts_node, _v_a_node)

                        label = self._trans_sys._graph.nodes[_u_ts_node].get('ap')
                        if self._trans_sys._graph.nodes[_u_ts_node].get("player") == "eve":
                            weight = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0].get('weight')
                        else:
                            weight = '0'
                        auto_label = self._auto_graph._graph.get_edge_data(_u_a_node, _v_a_node)[0]['guard']

                        if self._trans_sys._graph.nodes[_u_ts_node].get('player') == 'eve':
                            if auto_label.formula == "(true)" or auto_label.formula == "1":
                                truth = True
                            else:
                                truth = auto_label.check(label)

                        # if the node belongs to adam
                        else:
                            _v_a_node = _u_a_node
                            if not absorbing:
                                _v_prod_node = self.composition(_v_ts_node, _v_a_node)
                            else:
                                if _v_a_node in self._auto_graph.get_absorbing_states():
                                    _v_prod_node = self.add_prod_state(_v_a_node, _v_a_node)
                                else:
                                    _v_prod_node = self.composition(_v_ts_node, _v_a_node)
                            truth = True

                        if truth:
                            if not self._graph.has_edge(_u_prod_node, _v_prod_node):
                                if _u_prod_node in self._auto_graph.get_absorbing_states():
                                    if self._auto_graph._graph.nodes[_u_prod_node].get('accepting'):
                                        self._graph.add_weighted_edges_from([(_u_prod_node,
                                                                              _v_prod_node, '0')])
                                    else:
                                        self._graph.add_weighted_edges_from([(_u_prod_node,
                                                                              _v_prod_node,
                                                                              max_w)])
                                                                              # str(-1 * float(max_w)))])
                                else:
                                    self._graph.add_weighted_edges_from([(_u_prod_node, _v_prod_node,
                                                                          weight)])
                                                                          # str(-1* float(weight)))])

    def add_prod_state(self, _p_node, auto_node) -> None:
        """
        A helper method which is called when we use the absorbing flag to add manually created node to the product graph
        """
        if not self._graph.has_node(_p_node):
            self._graph.add_node(_p_node)
            # self._graph.nodes[_p_node]['ap'] = self._trans_sys._graph.nodes[ts_node].get('ap')

            if self._auto_graph._graph.nodes[auto_node].get('accepting'):
                self._graph.nodes[_p_node]['accepting'] = True

        return _p_node

    def composition(self, ts_node, auto_node) -> Tuple:
        _p_node = (ts_node, auto_node)

        if not self._graph.has_node(_p_node):
            self._graph.add_node(_p_node)
            self._graph.nodes[_p_node]['ts'] = ts_node
            self._graph.nodes[_p_node]['dfa'] = auto_node
            self._graph.nodes[_p_node]['ap'] = self._trans_sys._graph.nodes[ts_node].get('ap')

            # self._graph.add_node(_p_node, ts=ts_node, dfa=auto_node, obs=self._trans_sys._graph.nodes[_p_node]['ap'])

            if (self._trans_sys._graph.nodes[ts_node].get('init') and
                    self._auto_graph._graph.nodes[auto_node].get('init')):
                # if both the transition node and the dfa node are belong to the initial state sets then set this
                # product node as initial too
                self._graph.nodes[_p_node]['init'] = True

            if self._auto_graph._graph.nodes[auto_node].get('accepting'):
                # if both the transition node and the dfa node are belong to the accepting state sets then set this
                # product node as initial too
                self._graph.nodes[_p_node]['accepting'] = True

            if self._trans_sys._graph.nodes[ts_node].get('player') == 'eve':
                self._graph.nodes[_p_node]['player'] = 'eve'
            else:
                self._graph.nodes[_p_node]['player'] = 'adam'

        return _p_node

    def prune_graph(self, debug: bool = False):
        # initialize queue (deque is part of std library and allows O(1) append() and pop() at either end)
        queue = deque()
        regions = defaultdict(lambda : -1)
        attr = []  # the attractor region
        eve_str = {}

        for node in self.get_accepting_states():
            queue.append(node)
            regions[node] = +1
            attr.append(node)

        while queue:
            _n = queue.popleft()

            for _pre_n in self._graph.predecessors(_n):
                if regions[_pre_n] == -1 or self._graph.nodes[_pre_n].get("player") == "eve":
                    if self._graph.nodes[_pre_n].get("player") == "adam":
                        queue.append(_pre_n)
                        regions[_pre_n] = +1
                        attr.append(_pre_n)
                    else:
                        if regions[_pre_n] == -1:
                            queue.append(_pre_n)
                            regions[_pre_n] = +1
                            attr.append(_pre_n)
                        if not eve_str.get(_pre_n):
                            eve_str.update({_pre_n: [_n]})
                        else:
                            eve_str[_pre_n].append(_n)

        # debug
        if debug:
            print("=====================================")
            init_node = self.get_initial_states()[0][0]
            if init_node in attr:
                print("A Winning Strategy may exists")
            else:
                print("A Winning Strategy does not exists at all")
            print("=====================================")

        nx.set_edge_attributes(self._graph, False, "prune")
        # lets prune the graph by removing edges of eve do not exist in eve_str
        for _u, _vs in eve_str.items():
            for _v in _vs:
                # add attribute for the corresponding edge and the remove edges without this particular attribut
                self._graph.edges[_u, _v, 0]["prune"] = True

        self.prune_edges(debug=debug)

        # do a sanity check to make sure the final graph is total indeed
        self._sanity_check(debug=debug)

    def _sanity_check(self, debug: bool = False):
        # check is the graph is total or not by loop through every node and add a self-loop of weight max(W)
        # to every node that does not  have a successor
        max_w : str = self._trans_sys.get_max_weight()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                if debug:
                    print("=====================================")
                    print(f"Adding self loop of weight - {max_w} to the node {_n}")
                    print("=====================================")
                self._graph.add_weighted_edges_from([(_n, _n, str(-1 * float(max_w)))])

    def prune_edges(self, debug):
        # A helper function to remove edges without the "prune" attribute
        remove_lst = []
        for _ed in self._graph.edges.data():
            if (not _ed[2].get("prune")) and self._graph.nodes[_ed[0]].get("player") == "eve":
                remove_lst.append(_ed)

        if debug:
            print("=====================================")
            print(f"The number of edges prunes are : {len(remove_lst)}")
            print("=====================================")

        for _e in remove_lst:
            if debug:
                print("=====================================")
                print(f"Removing edge between {_e[0]}--->{_e[1]}")
                print("=====================================")
            self._graph.remove_edge(_e[0], _e[1])


class ProductBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self, **kwargs):
        pass