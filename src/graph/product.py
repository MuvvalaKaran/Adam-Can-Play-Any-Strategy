import sys
import math
import copy
import yaml
import queue
import warnings
import networkx as nx
import numpy as np

from typing import List, Tuple, Dict, Set, Optional, Union
from collections import deque, defaultdict
from graphviz import Digraph

# import local packages
from .base import Graph
from .two_player_graph import TwoPlayerGraph
from .trans_sys import FiniteTransSys
from .dfa import DFAGraph
from .pdfa import PDFAGraph
from ..factory.builder import Builder
from ..spot.Parser import parse as parse_guard

PROD_ACCEPTING_STATE_NAME = 'Accepting'
AUTO_ACCEPTING_STATE_NAME = 'qA'
TS_INIT_STATE_NAME = 'Init'
TS_EXTRA_STATE_NAME = 'tA'


class ProductAutomaton(TwoPlayerGraph):

    def __init__(self,
                 trans_sys: Optional[TwoPlayerGraph],
                 automaton: Union[DFAGraph, PDFAGraph],
                 graph_name: str,
                 config_yaml,
                 save_flag: bool = False,
                 absorbing: bool = False,
                 finite: bool = False,
                 plot_auto_graph: bool = False,
                 plot_trans_graph: bool = False,
                 alpha: float = 1.0,
                 observe_next_on_trans: bool = True,
                 integrate_accepting: bool = False,
                 use_trans_sys_weights: bool = False,
                 pdfa_compose: bool = False,
                 skip_empty: bool = True) -> 'ProductAutomaton()':

        self._trans_sys: Optional[TwoPlayerGraph] = copy.deepcopy(trans_sys)
        self._auto_graph: Union[DFAGraph, PDFAGraph] = copy.deepcopy(automaton)
        self._transitions = defaultdict(lambda: defaultdict(lambda: None))

        self._absorbing = absorbing

        self._plot_auto_graph = plot_auto_graph
        self._plot_trans_graph = plot_trans_graph

        self._observe_next_on_trans = observe_next_on_trans
        self._integrate_accepting = integrate_accepting
        self._use_trans_sys_weights = use_trans_sys_weights

        self._pdfa_compose: bool = pdfa_compose

        ts_node_default_attr = {'ap': set()}
        ts_edge_default_attr = {'actions': '', 'weight': 0, 'weights': None}
        au_node_default_attr = {}
        au_edge_default_attr = {'guard': parse_guard('1'), 'weight': 0, 'prob': 1}
        pd_edge_default_attr = {'actions': '', 'weights': [0, 0]}
        self._default_attr = {
            'ts_node': ts_node_default_attr,
            'ts_edge': ts_edge_default_attr,
            'au_node': au_node_default_attr,
            'au_edge': au_edge_default_attr,
            'pd_edge': pd_edge_default_attr}

        if isinstance(automaton, PDFAGraph):
            self._multiple_weights = True
        else:
            self._multiple_weights = False

        self._skip_empty = skip_empty

        TwoPlayerGraph.__init__(self,
                                graph_name=graph_name,
                                config_yaml=config_yaml,
                                save_flag=save_flag,
                                finite=finite)

    def get_attr(self, key: str, attr_name: str, attr_dict: Dict):
        if key not in self._default_attr:
            raise Exception(f'{key} not found in default attribute')

        default_attr = self._default_attr[key]
        return self._get_attr(attr_name, attr_dict, default_attr)

    def _get_attr(self, attr_name: str, attr_dict: Dict, default_dict: Dict):
        if attr_name in attr_dict:
            return attr_dict[attr_name]
        elif attr_name in default_dict:
            return default_dict[attr_name]
        else:
            return None

    def compose_graph(self):

        # throw a warning if the DFA does NOT contain any symbol that appears in the set of observations in TS
        # if not self._check_ts_ltl_compatability():
        #     warnings.warn("Please make sure that the formula is composed of symbols that are part of the aps in the TS")
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        # This is the product construction method used by Kandai in fpr his PDFA product construction
        # TODO: Add confunctionality to construct non-absorbing product automaton for non-pdfa instances too!
        if self._pdfa_compose:
            self._extend_trans_init()

            if self._absorbing:

                # if you want to construct pdfa product automaton
                self.construct_minimum_product()

                # all the edges from the sys node to the absorbing state should have weight zero.
                if not isinstance(self._auto_graph, PDFAGraph):
                    self._add_sys_to_abs_states_w_zero_wgt()

            else:
                self.construct_product()

            if self._integrate_accepting:
                self._integrate_accepting_states()

            self._sanity_check(debug=True)

            self._initialize_edge_labels_on_fancy_graph()
            if self._config_yaml is not None:
                self.dump_to_yaml()
        else:
            self.construct_product_absorbing()
            
            self._sanity_check(debug=False)

    def _extend_trans_init(self):
        # Get the original initial state
        _n = self._trans_sys.get_initial_states()[0][0]
        # Set its init to False
        self._trans_sys._graph.nodes[_n]['init'] = False

        player = self._trans_sys._graph.nodes[_n]['player']
        actions = f'{TS_INIT_STATE_NAME}_to_{_n}'

        # Add an artificial initial state
        self._trans_sys.add_state(TS_INIT_STATE_NAME, init=True, player=player, ap='')

        # Add an edge from this state to the original init state
        self._trans_sys.add_edge(TS_INIT_STATE_NAME, _n,
                                 weight=0,
                                 actions=actions)

        if self._plot_trans_graph:
            orig_name = self._trans_sys._graph.name
            self._trans_sys._graph.name = orig_name + '_extended'
            self._trans_sys.plot_graph()

    def _integrate_accepting_states(self):
        # if not isinstance(self._auto_graph, PDFAGraph):
        #     return

        for _n in self.get_accepting_states():
            # Check if absorbing, then delete
            # TODO: What if the absorbing state is also an init state
            _preds = [_m for _m in self._graph.predecessors(_n)]

            self._graph.nodes[_n]['accepting'] = False
            self._graph.nodes[_n]['originalAccepting'] = True
            prob = self._graph.nodes[_n].get('final_probability')
            weight = 0.0 if prob in [1.0, None] else -math.log(prob)

            player = self._trans_sys._graph.nodes[_n[0]].get('player')
            # Turn node's accepting to False
            # Add an edge between node to the accepting state
            self.add_edge(_n, PROD_ACCEPTING_STATE_NAME,
                        weight=weight,
                        actions=set([f'toAcceptingBy{player}']),
                        player=player,
                        weights={'ts': weight,'pref': 0},
                        pref=1)
            self._transitions[_n][f'toAcceptingBy{player}'] = PROD_ACCEPTING_STATE_NAME

        self.add_state(PROD_ACCEPTING_STATE_NAME,
                       ts=None, dfa=None, player='eve', accepting=True, ap='')

    def _check_ts_ltl_compatability(self) -> bool:
        """
        Return true if the DFA contains atleast one symbols that is part of the set of observations is TS else False
        :return: True if the automation indeed is constructed based on symbols from the TS else False
        """

        dfa_symbols: List[str] = self._auto_graph.get_symbols()
        ts_aps: Set[str] = self._trans_sys._get_set_ap()

        flag = True
        for sym in dfa_symbols:
            if sym not in ts_aps:
                print(f"Symbol {sym} is not part of the set of aps in the Transition System"
                      f" {self._trans_sys._graph_name}")
                flag = False

        return flag

    def construct_minimum_product(self):
        # Create an product initial state
        t_init = self._trans_sys.get_initial_states()[0][0]
        q_init = self._auto_graph.get_initial_states()[0][0]
        _n_init = self._composition(t_init, q_init)

        # Create Queues
        searchQueue = queue.Queue()
        visited = defaultdict(lambda: False)

        searchQueue.put(_n_init)
        visited[_n_init] = True

        # Create transitions until there is no more
        while not searchQueue.empty():
            _u_prod_node = searchQueue.get()

            _u_ts_node = _u_prod_node[0]
            _u_a_node = _u_prod_node[1]

            # Next we check for all transitions in TS
            for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):

                # Get all info about the TS transition
                ts_edge_attrs = self._trans_sys._graph[_u_ts_node][_v_ts_node]
                n = _v_ts_node if self._observe_next_on_trans else _u_ts_node
                ap = self._trans_sys._graph.nodes[n].get('ap')
                # ts_action, ap, weight = self._get_ts_transition_data(_u_ts_node, _v_ts_node)
                # ts_weights = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0]\
                #     .get('weights')

                # Check if the trans in TS satisfies any trans in the Automaton specification
                for _v_a_node in self._auto_graph._graph.successors(_u_a_node):
                    _v_prod_node = (_v_ts_node, _v_a_node)

                    # Assume multiple edges
                    # auto_weights, prefs, auto_actions = self._get_auto_transition_data(
                    #     _u_a_node, _v_a_node)
                    auto_edge_attrs = self._auto_graph._graph[_u_a_node][_v_a_node]

                    # For each symbol, check if ap satisfies any of them
                    # for auto_weight, pref, auto_action in zip(auto_weights, prefs, auto_actions):
                    for auto_edge_attr in auto_edge_attrs.values():

                        # determine if a transition is possible or not, if yes then add that edge
                        auto_action = self.get_attr('au_edge', 'guard', auto_edge_attr)
                        # pref = self.get_attr('au_edge', 'prob', auto_edge_attr)
                        auto_weight = self.get_attr('au_edge', 'weight', auto_edge_attr)

                        exists, _ = self._check_transition_absorbing(
                            _u_ts_node, _v_ts_node,
                            _u_a_node, _v_a_node,
                            _v_prod_node, action=auto_action, obs=ap)

                        # print(exists, auto_action, ap)

                        if not exists:
                            continue

                        if not visited[_v_prod_node]:
                            self._composition(_v_ts_node, _v_a_node)
                            # check if it's already been visited
                            searchQueue.put(_v_prod_node)
                            visited[_v_prod_node] = True

                        for ts_edge_attr in ts_edge_attrs.values():
                            weight = self.get_attr('ts_edge', 'weight', ts_edge_attr)
                            ts_action = self.get_attr('ts_edge', 'actions', ts_edge_attr)

                            # if self._skip_empty and len(ap) == 0:
                            #     auto_weight = 0

                            if self._use_trans_sys_weights:
                                weights = self.get_attr('ts_edge', 'weights', ts_edge_attr)
                            else:
                                weights = {'ts': weight,'pref': auto_weight}

                            self._add_transition_absorbing(_u_prod_node,
                                                            _v_prod_node,
                                                            weight=weight,
                                                            action=ts_action,
                                                            weights=weights,
                                                            # pref=pref,
                                                            ap=ap)

    def construct_product(self):
        """
        A function that helps build the composition of TS and DFA. Unlike the absorbing case, the accepting states and
        the set of trap states are all not compressed into one.

        :return: The composed graph
        """

        for _u_ts_node in self._trans_sys._graph.nodes():
            for _u_a_node in self._auto_graph._graph.nodes():

                _u_prod_node = self._composition(_u_ts_node, _u_a_node)

                for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):

                    ts_action, ap, weight = self._get_ts_transition_data(_u_ts_node, _v_ts_node)

                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):
                        _v_prod_node = self._composition(_v_ts_node, _v_a_node)

                        # Assume multiple edges
                        auto_weights, prefs, auto_actions = self._get_auto_transition_data(
                            _u_a_node, _v_a_node)

                        for auto_weight, pref, auto_action in zip(auto_weights, prefs, auto_actions):
                            # determine if a transition is possible or not, if yes then add that edge
                            exists, _v_prod_node = self._check_transition(_u_ts_node,
                                                                        _v_ts_node,
                                                                        _u_a_node,
                                                                        _v_a_node,
                                                                        _v_prod_node,
                                                                        action=auto_action,
                                                                        obs=ap,)
                                                                        # pref=pref)

                            if exists:
                                self._add_transition(_u_prod_node,
                                                    _v_prod_node,
                                                    weight=weight,
                                                    action=ts_action,
                                                    weights={'ts': weight,'pref': auto_weight})

    def construct_product_absorbing(self):
        """
        A function that helps build the composition of TS and DFA where we compress the all
        absorbing nodes (A node that only has a transition to itself) in product automation

        There are always two of these - one accepting state and the other one is the "trap" state
        :return:
        """
        # get max weight from the transition system
        max_w: float = self._trans_sys.get_max_weight()

        for _u_ts_node in self._trans_sys._graph.nodes():
            for _u_a_node in self._auto_graph._graph.nodes():

                # if the current node an absorbing state, then we don't compose u_ts_node and u_a_node
                if _u_a_node in self._auto_graph.get_absorbing_states():
                    _u_prod_node = self._add_prod_state(_u_a_node, _u_a_node)
                else:
                    _u_prod_node = self._composition(_u_ts_node, _u_a_node)

                for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):

                    ts_action, ap, weight = self._get_ts_transition_data(_u_ts_node, _v_ts_node)

                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):

                        # if the next node is an absorbing state, then we don't compose v_ts_node and v_a_node
                        if _v_a_node in self._auto_graph.get_absorbing_states():
                            _v_prod_node = self._add_prod_state(_v_a_node, _v_a_node)
                        else:
                            _v_prod_node = self._composition(_v_ts_node, _v_a_node)

                        # Assume multiple edges
                        auto_weights, prefs, auto_actions = self._get_auto_transition_data(
                            _u_a_node, _v_a_node)

                        for auto_weight, pref, auto_action in zip(auto_weights, prefs, auto_actions):
                            exists, _v_prod_node = self._check_transition_absorbing(_u_ts_node,
                                                                                    _v_ts_node,
                                                                                    _u_a_node,
                                                                                    _v_a_node,
                                                                                    _v_prod_node,
                                                                                    action=auto_action,
                                                                                    obs=ap,)
                                                                                    # pref=pref)
                            if exists:
                                self._add_transition_absorbing(_u_prod_node,
                                                              _v_prod_node,
                                                              weight=weight,
                                                              action=ts_action,
                                                              weights={'ts': weight,'pref': auto_weight})

    def _add_transition(self, _u_prod_node, _v_prod_node, weight: float, action: str,
                        **kwargs):
        """
        A helper method to add an edge if it already does not exists given the current product node
        (composition of current TS and current DFA node) and the next product node.

        This method is invoked when the absrobing flag is false. We do not represent all absorbing
        (accepting and trap states) as a single state
        :param _u_prod_node:
        :param _v_prod_node:
        :return: Updated the graph with the edge
        """

        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            self.add_edge(_u_prod_node, _v_prod_node,
                          weight=weight,
                          actions=action,
                          **kwargs)

    def _add_transition_absorbing(self,
                                  _u_prod_node,
                                  _v_prod_node,
                                  weight: float,
                                  action: str,
                                  **kwargs) -> None:
        """
        A helper method to add an edge if it already does not exists given the current product node
        (composition of current TS and current DFA node) and the next product node.

        This method is invoked when the absorbing flag is True. In this method absorbin
        states are manually added a self loop of weight 0

        :param _u_prod_node:
        :param _v_prod_node:
        :return: Updated graph with the edge
        """
        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            if _u_prod_node in self._auto_graph.get_absorbing_states():
                # absorbing state
                self.add_edge(_u_prod_node, _v_prod_node,
                                weight=0,
                                actions=action,
                                **kwargs)
            else:
                self.add_edge(_u_prod_node, _v_prod_node,
                                weight=weight,
                                actions=action,
                                **kwargs)
            if isinstance(action, str):
                self._transitions[_u_prod_node][action] = _v_prod_node
            else:
                for a in action:
                    self._transitions[_u_prod_node][a] = _v_prod_node

    def _add_sys_to_abs_states_w_zero_wgt(self):
        """
        A method that computes the set of all accepting and trap states (jointly the absorbing states), identifies the
        sys nodes that only transit to these states, and manually overwrite the org weight to have an transition of
        weight 0.

        Originally the edge weights are carried over by the self-loop/transition at these states from the Transition
        system. When we take the product, the weights simply get transferred. As we consider the current state
        observation while constructing the  product, an edge transits from the sys node to these absorbing states
        :return:
        """
        # add nodes that have direct edges to the absorbing state as zero
        _accp_state = self.get_accepting_states()
        _trap_state = self.get_trap_states()

        # as absorbing states have self-loops, we also look at the self-loop weight and override it to be 0,
        for _s in _accp_state + _trap_state:
            for _pre_s in self._graph.predecessors(_s):
                self._graph[_pre_s][_s][0]['weight'] = 0

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
        # we force the next node in automaton to be the current node and add that transition
        else:
            _v_a_node = _u_a_node
            _v_prod_node = self._composition(_v_ts_node, _v_a_node)
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
        # we force the next node in automaton to be the current node and add that transition
        elif self._trans_sys._graph.nodes[_u_ts_node].get('player') == 'adam':
            # TODO: Do not carry eve's action to adam's node
            # _v_a_node = _u_a_node

            # # if the next node belongs to an absorbing state
            # if _v_a_node in self._auto_graph.get_absorbing_states():
            #     _v_prod_node = self._add_prod_state(_v_a_node, _v_a_node)

            # else:
            #     _v_prod_node = self._composition(_v_ts_node, _v_a_node)

            # TODO: Ask Karan why this statement is always True.
            # return True, _v_prod_node
            if action.formula == "(true)" or action.formula == "1":
                return True, _v_prod_node
            else:
                 return action.check(obs), _v_prod_node

        else:
            warnings.warn(f"Looks like the node {_u_ts_node} in graph {self._trans_sys._graph_name} does"
                          f" not have a valid player assigned to it")

    def _get_ts_transition_data(self, _u_ts_node, _v_ts_node) -> Tuple[str, str, float]:
        # Assumes only one edge exists betw. nodes in TS

        _ts_action = self._trans_sys.get_edge_attributes(_u_ts_node, _v_ts_node, 'actions')

        node = _v_ts_node if self._observe_next_on_trans else _u_ts_node
        obs = self._trans_sys._graph.nodes[node].get('ap')

        try:
            _weight = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0].get('weight')
        except:
            warnings.warn(f"The edge from {_u_ts_node} to {_v_ts_node} does not contain the attribute 'weight'."
                          f"Setting the edge weight to 0 while constructing the product")
            _weight = 0

        return _ts_action, obs, _weight

    def _get_auto_transition_data(self, _u_auto_node, _v_auto_node) -> Tuple[List, List]:
        auto_edge_data = self._auto_graph._graph.get_edge_data(_u_auto_node, _v_auto_node)
        num_auto_edges = len(auto_edge_data)

        prefs = []
        weights = []
        automaton_labels = []

        for i_edge in range(num_auto_edges):

            automaton_label = auto_edge_data[i_edge]['guard']

            if self._multiple_weights:
                try:
                    weight = auto_edge_data[i_edge].get('weight')
                    pref = auto_edge_data[i_edge].get('prob')
                except:
                    msg = f"The weight from edge {_u_auto_node} to {_v_auto_node} does not exist"
                    warnings.warn(msg)
                    weight = 0
                    pref = 1
            else:
                weight = 0
                pref = 1

            weights.append(weight)
            prefs.append(pref)
            automaton_labels.append(automaton_label)

        return weights, prefs, automaton_labels

    def _add_prod_state(self, _p_node, auto_node) -> Tuple:
        """
        A helper method which is called when we use the absorbing flag to add manually
         created node to the product graph
        """
        if not self._graph.has_node(_p_node):
            self._graph.add_node(_p_node)
            # self._graph.nodes[_p_node]['ap'] = self._trans_sys._graph.nodes[ts_node].get('ap')

            if self._auto_graph._graph.nodes[auto_node].get('accepting'):
                self._graph.nodes[_p_node]['accepting'] = True
            
            self._graph.nodes[_p_node]['player'] = 'eve'

        return _p_node

    def _composition(self, ts_node, auto_node) -> Tuple:
        _p_node = (ts_node, auto_node)

        if not self._graph.has_node(_p_node):
            self._graph.add_node(_p_node)
            self._graph.nodes[_p_node]['ts'] = ts_node
            self._graph.nodes[_p_node]['dfa'] = auto_node
            self._graph.nodes[_p_node]['ap'] = self._trans_sys._graph.nodes[ts_node].get('ap')

            if (self._trans_sys._graph.nodes[ts_node].get('init') and
                    self._auto_graph._graph.nodes[auto_node].get('init')):
                # if both the transition node and the dfa node are belong to the initial state sets then set this
                # product node as initial too
                self._graph.nodes[_p_node]['init'] = True

            if self._auto_graph._graph.nodes[auto_node].get('accepting'):
                # if the dfa node belongs to the set of accepting states then set this product node as accepting too
                self._graph.nodes[_p_node]['accepting'] = True
                prob = self._auto_graph._graph.nodes[auto_node].get('final_probability')
                if prob:
                    self._graph.nodes[_p_node]['final_probability'] = prob

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

        self._prune_edges(debug=debug)

        # do a sanity check to make sure the final graph is total indeed
        self._sanity_check(debug=debug)

    def _sanity_check(self, debug: bool = False):
        if self._finite:
            return
        # check is the graph is total or not by looping through every nodE and add a self-loop of weight max(W)
        # to every node that does not  have a successor
        max_w: float = self._trans_sys.get_max_weight()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                player = self.get_state_w_attribute(_n, 'player')
                if debug:
                    print("=====================================")
                    print(f"Adding self loop of weight - {max_w} to the node {_n}")
                    print("=====================================")
                # self._graph.add_weighted_edges_from([(_n, _n, 0)])
                self.add_edge(_n, _n,
                            weight=0,
                            actions=f'loopBy{player}',
                            weights={'ts': 0,'pref': 0},
                            pref=1)

    def _prune_edges(self, debug):
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

    def set_node_labels_on_fancy_graph(self, labels: Dict):
        """
        :arg labels:    A dict of nodes to labels
        """
        for node, label in labels.items():
            self.add_state_attribute(node, 'label', label)

    def set_edge_labels_on_fancy_graph(self, labels: Dict):
        """
        :arg labels:    A dict of edges to labels
        """
        for edge, label in labels.items():
            u = edge[0]
            v = edge[1]
            self._graph[u][v][0]['label'] = label

    def _initialize_edge_labels_on_fancy_graph(self, round_float_by: int = 2):

        edges = set(self._graph.edges())

        for edge in edges:
            for i, edge_data in self._graph[edge[0]][edge[1]].items():
                actions = self.get_attr('pd_edge', 'actions', edge_data)
                weights = self.get_attr('pd_edge', 'weights', edge_data)
                # actions = self.get_edge_attributes(edge[0], edge[1], 'actions')
                # weights = self.get_edge_attributes(edge[0], edge[1], 'weights')
                label = copy.deepcopy(weights)
                label.update({'actions': actions})
                self._graph[edge[0]][edge[1]][i]['label'] = str(label)

        # self.set_edge_labels_on_fancy_graph(edge_labels)

    def set_strategy(self, edges: List):
        for edge in edges:
            u = edge[0]
            v = edge[1]
            self._graph[u][v][0]['strategy'] = True

    def fancy_graph(self, color=("lightgrey", "red", "purple", "cyan"), **kwargs) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            xlabel = n[1].get('label')
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "xlabel": xlabel,
                                             "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1], "xlabel": xlabel})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2], "xlabel": xlabel})
            if n[1].get('originalAccepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[3], "xlabel": xlabel})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "circle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = edge[2].get('label')

            if edge[2].get('strategy') is True:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')), _attributes={'color': 'red'})
                dot.edge(str(edge[0]), str(edge[1]), label=label, _attributes={'color': 'red'})
            else:
                # dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('weight')))
                dot.edge(str(edge[0]), str(edge[1]), label=label)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, **kwargs)

    def reset(self):
        virtual_init_node = self.get_initial_states()[0][0]
        sys_state = list(self._graph.successors(virtual_init_node))[0]
        env_action = self.get_edge_attributes(virtual_init_node, sys_state, 'actions')
        weights = self.get_edge_attributes(virtual_init_node, sys_state, 'weights')

        # return virtual_init_node, env_action, sys_state, list(weights.values())
        return None, None, virtual_init_node, [0, 0]

    def next_transition(self, state, action):

        if state not in self._transitions:
            raise Exception(f'{state} not in the graph')
        
        # if action is of tuple like ('north', 'north') then reconstruct it
        # to north_north_None if state is sys else None_south_south
        if self.get_state_w_attribute(state, 'player') == 'eve':
            action = '_'.join(action)
            action += '__None'
        elif self.get_state_w_attribute(state, 'player') == 'adam':
            action = 'None__' + '_'.join(action)
        else: 
            warnings.warn(f"Came across a Minigrid abstraction states: {state} with no player assigned to it.")
            sys.exit(-1)


        if action not in self._transitions[state]:
            actions = list(self._transitions[state].keys())

            raise Exception(f'{action} not in {actions} at {state}')

        return self._transitions[state][action]

    def step(self, state, action, sys_chosen_next_state = None):
        next_state = self.next_transition(state, action)

        if sys_chosen_next_state is not None and next_state != sys_chosen_next_state:
            next_state = sys_chosen_next_state
            warnings.warn(f'Chose state {sys_chosen_next_state} over {next_state}.')

        weights = self.get_edge_attributes(state, next_state, 'weights')
        obs = self._graph[state][next_state][0].get('ap', '')

        # done = self._graph.nodes[next_state].get('originalAccepting')
        done = self._graph.nodes[next_state].get('accepting')
        done = False if done is None else True

        return next_state, obs, list(weights.values()), done


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
                 prune: bool = False,
                 debug: bool = False,
                 plot: bool = False,
                 view: bool = True,
                 format: str = 'pdf',
                 from_file: bool = False,
                 **kwargs):
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

        self._instance = ProductAutomaton(**kwargs)

        graph_yaml = None
        if 'config_yaml' in kwargs and kwargs['config_yaml'] is not None and from_file:
            graph_yaml = self._from_yaml(kwargs['config_yaml'])

        if graph_yaml is not None:
            self._instance.construct_graph(graph_yaml)
        else:
            trans_sys = kwargs['trans_sys'] if 'trans_sys' in kwargs else None
            automaton = kwargs['automaton'] if 'automaton' in kwargs else None

            if trans_sys is not None and automaton is not None:
                self._instance.compose_graph()

            if prune:
                self._instance.prune_graph(debug=debug)

        if plot:
            self._instance.plot_graph(view=view, format=format)

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:
        config_data = self.load_YAML_config_data(config_file_name)

        return config_data
