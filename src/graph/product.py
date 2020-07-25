import networkx as nx
import warnings
import math

from typing import List, Tuple, Dict, Set
from collections import deque, defaultdict

# import local packages
from .base import Graph
from .two_player_graph import TwoPlayerGraph
from .trans_sys import FiniteTransSys
from .dfa import DFAGraph
from src.factory.builder import Builder


class ProductAutomaton(TwoPlayerGraph):

    def __init__(self,
                 trans_sys_graph: Graph,
                 automaton: DFAGraph,
                 graph_name: str,
                 config_name,
                 save_flag: bool = False) -> 'ProductAutomaton()':
        self._trans_sys = trans_sys_graph
        self._auto_graph = automaton
        TwoPlayerGraph.__init__(self,
                                graph_name=graph_name,
                                config_yaml=config_name,
                                save_flag=save_flag)

    def construct_graph(self):
        super().construct_graph()

    def compose_graph(self, absorbing: bool = False):

        # throw a warning if the DFA does NOT contain any symbol that appears in the set of observations in TS
        if not self._check_ts_ltl_compatability():
            warnings.warn("Please make sure that the formula is composed of symbols that are part of the aps in the TS")

        if absorbing:
            self.construct_product_absorbing()

        else:
            self.construct_product()

    def _check_ts_ltl_compatability(self) -> bool:
        """
        Return true if the DFA contains atleast one symbols that is part of the set of observations is TS else False
        :return: True if the automation indeed is constructed based on symbols from the TS else False
        """

        dfa_symbols: List[str] = self._auto_graph._get_symbols()
        ts_aps: Set[str] = self._trans_sys._get_set_ap()

        flag = True
        for sym in dfa_symbols:
            if sym not in ts_aps:
                print(f"Symbol {sym} is not part of the set of aps in the Transition System"
                      f" {self._trans_sys._graph_name}")
                flag = False

        return flag

    def construct_product(self):
        """
        A function that helps build the composition of TS and DFA
        :return: The composed graph
        """

        for _u_ts_node in self._trans_sys._graph.nodes():
            for _u_a_node in self._auto_graph._graph.nodes():

                _u_prod_node = self.composition(_u_ts_node, _u_a_node)

                for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):
                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):
                        _v_prod_node = self.composition(_v_ts_node, _v_a_node)

                        # get relevant details that need to be added to the composed node and edge
                        ap, weight, action = self._get_edge_and_node_data(_u_ts_node,
                                                                          _v_ts_node,
                                                                          _u_a_node,
                                                                          _v_a_node)

                        # determine if a transition is possible or not, if yes then add that edge
                        exists, _v_prod_node = self._check_transition(_u_ts_node,
                                                                      _v_ts_node,
                                                                      _u_a_node,
                                                                      _v_a_node,
                                                                      _v_prod_node,
                                                                      action=action,
                                                                      obs=ap)

                        if exists:
                            self._add_transition(_u_prod_node, _v_prod_node, weight=weight)

    def construct_product_absorbing(self):
        """
        A function that helps build the composition of TS and DFA where we compress the all
         absorbing nodes (A node that only has a transition to itself) in product automation

         There are always two these - one accepting node and the other one is the "trap" state
        :return:
        """
        # get max weight from the transition system
        max_w: str = self._trans_sys.get_max_weight()

        for _u_ts_node in self._trans_sys._graph.nodes():
            for _u_a_node in self._auto_graph._graph.nodes():

                # if the current node an absorbing state, then we don't compose u_ts_node and u_a_node
                if _u_a_node in self._auto_graph._get_absorbing_states():
                    _u_prod_node = self.add_prod_state(_u_a_node, _u_a_node)

                else:
                    _u_prod_node = self.composition(_u_ts_node, _u_a_node)

                for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):
                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):

                        # if the next node is an absorbing state, then we don't compose v_ts_node and v_a_node
                        if _v_a_node in self._auto_graph._get_absorbing_states():
                            _v_prod_node = self.add_prod_state(_v_a_node, _v_a_node)
                        else:
                            _v_prod_node = self.composition(_v_ts_node, _v_a_node)

                        # get relevant details that need to be added to the composed node and edge
                        ap, weight, action = self._get_edge_and_node_data(_u_ts_node,
                                                                          _v_ts_node,
                                                                          _u_a_node,
                                                                          _v_a_node)

                        exists, _v_prod_node = self._check_transition_absorbing(_u_ts_node,
                                                                                _v_ts_node,
                                                                                _u_a_node,
                                                                                _v_a_node,
                                                                                _v_prod_node,
                                                                                action=action,
                                                                                obs=ap)
                        if exists:
                            self._add_transition_absorbing(_u_prod_node,
                                                           _v_prod_node,
                                                           weight=weight,
                                                           max_weight=max_w)

    def _add_transition(self, _u_prod_node, _v_prod_node, weight: str):
        """
        A helper method to add an edge if it alredy does not exists given the current product node
        (composition of current TS and current DFA node) and the next product node.
        :param _u_prod_node:
        :param _v_prod_node:
        :return: Updated the graph with the edge
        """

        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            self._graph.add_weighted_edges_from([(_u_prod_node,
                                                  _v_prod_node,
                                                  weight)])

    def _add_transition_absorbing(self, _u_prod_node, _v_prod_node, weight: str, max_weight: str):

        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            if _u_prod_node in self._auto_graph._get_absorbing_states():
                if self._auto_graph._graph.nodes[_u_prod_node].get('accepting'):
                    self._graph.add_weighted_edges_from([(_u_prod_node,
                                                          _v_prod_node, '0')])
                else:
                    self._graph.add_weighted_edges_from([(_u_prod_node,
                                                          _v_prod_node,
                                                          max_weight)])
                                                          # str(-1 * float(max_weight)))])
            else:
                self._graph.add_weighted_edges_from([(_u_prod_node, _v_prod_node,
                                                      weight)])
                                                      # str(-1 * float(weight)))])

    def _check_transition(self, _u_ts_node,
                          _v_ts_node,
                          _u_a_node,
                          _v_a_node,
                          _v_prod_node,
                          action,
                          obs) -> Tuple[bool, Tuple]:
        """
        A helper method that check if a transition between the composed nodes exists or not.
        :param _u_ts_node:
        :param _v_ts_node:
        :param action:
        :param obs:
        :param _v_a_node:
        :param _u_a_node:
        :return: True if there is a transition else False
        """

        # if the current node in TS belongs to eve
        if self._trans_sys._graph.nodes[_u_ts_node].get('player') == 'eve':
            if action.formula == "(true)" or action.formula == "1":
                return True, _v_prod_node
            else:
                return action.check(obs), _v_prod_node

        # if the current node in TS belongs to adam
        else:
            # we force the next node in automaton to be the current node and add that transition
            _v_a_node = _u_a_node
            _v_prod_node = self.composition(_v_ts_node, _v_a_node)
            return True, _v_prod_node

    def _check_transition_absorbing(self, _u_ts_node,
                                    _v_ts_node,
                                    _u_a_node,
                                    _v_a_node,
                                    _v_prod_node,
                                    action,
                                    obs) -> Tuple[bool, Tuple]:
        """
        A helper method that check if a transition between the composed nodes exists or not.
        :param _u_ts_node:
        :param _v_ts_node:
        :param _v_a_node:
        :param _u_a_node:
        :param action:
        :param obs:
        :return: True if there is a transition else False
        """

        # if the current node in TS belongs to eve
        if self._trans_sys._graph.nodes[_u_ts_node].get('player') == 'eve':
            if action.formula == "(true)" or action.formula == "1":
                return True, _v_prod_node
            else:
                return action.check(obs), _v_prod_node

        # if the current node in TS belongs to adam
        else:
            # we force the next node in automaton to be the current node and add that transition
            _v_a_node = _u_a_node
            # if the next node belongs to an absorbing state
            if _v_a_node in self._auto_graph._get_absorbing_states():
                _v_prod_node = self.add_prod_state(_v_a_node, _v_a_node)

            else:
                _v_prod_node = self.composition(_v_ts_node, _v_a_node)

            return True, _v_prod_node

    def _get_edge_and_node_data(self, _u_ts_node, _v_ts_node, _u_auto_node, _v_auto_node) -> Tuple:
        """
        A helper method that returns

            1. observation at the current node in TS (i.e at -observation at _u_ts_node)
            2. weight : The edges weight from the TS between _u_ts_node and _v_ts_node
                NOTE: all edges that belong to the env(adam/human) have edge weight zero
            3. transition label : The transition label from the automation between _u_auto_node and _v_auto_node
        :param _u_ts_node: The current node in TS
        :param _v_ts_node:  Neighbour/successor of the current node in TS
        :param _u_auto_node: The current node in Automation - DFA
        :param _v_auto_node: Neighbour/successor of the current node in DFA
        :return: A tuple of observation, weight, and transition label
        """

        obs = self._trans_sys._graph.nodes[_u_ts_node].get('ap')

        if self._trans_sys._graph.nodes[_u_ts_node].get("player") == "eve":
            weight = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0].get('weight')
        else:
            weight = '0'

        auto_label = self._auto_graph._graph.get_edge_data(_u_auto_node, _v_auto_node)[0]['guard']

        return obs, weight, auto_label

    def add_prod_state(self, _p_node, auto_node) -> Tuple:
        """
        A helper method which is called when we use the absorbing flag to add manually
         created node to the product graph
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
        regions = defaultdict(lambda: -1)
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
        max_w: str = self._trans_sys.get_max_weight()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                if debug:
                    print("=====================================")
                    print(f"Adding self loop of weight - {max_w} to the node {_n}")
                    print("=====================================")
                self._graph.add_weighted_edges_from([(_n,
                                                      _n,
                                                      math.inf)])
                # str(-1 * float(max_w)))])

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
    """
    Implements the generic graph builder class for TwoPlayerGraph
    """

    def __init__(self) -> 'ProductBuilder()':
        """
        Constructs a new instance of the Product automaton builder
        """
        Builder.__init__(self)

    def __call__(self,
                 graph_name: str,
                 config_yaml: str,
                 trans_sys: Graph = None,
                 dfa: DFAGraph = None,
                 save_flag: bool = False,
                 prune: bool = False,
                 debug: bool = False,
                 absorbing: bool = False,
                 plot: bool = False):
        """
        A function that takes as input a

            1. Transition system of type TwoPlayerGraph (Manually constructed TS) or
            FiniteTransitionSystem (constructed using a raw TS (with only system nodes))
            2. DFA - Automaton representing a ltl formula build using SPOT

        effect : Composes trans_sys and dfa and returns the concrete instace of ProductAutomaton
        :param graph_name:
        :param config_yaml:
        :param trans_sys:
        :param dfa:
        :param save_flag:
        :param prune:
        :param debug:
        :param absorbing:
        :return: A concrete and active instance of ProductAutomaton that is the composition TS and DFA
        """

        self._instance = ProductAutomaton(trans_sys_graph=trans_sys,
                                          automaton=dfa,
                                          graph_name=graph_name,
                                          config_name=config_yaml,
                                          save_flag=save_flag)

        if trans_sys is not None and dfa is not None:
            self._instance.construct_graph()
            self._instance.compose_graph(absorbing=absorbing)

        if prune:
            self._instance.prune_graph(debug=debug)

        if plot:
            self._instance.plot_graph()

        return self._instance
