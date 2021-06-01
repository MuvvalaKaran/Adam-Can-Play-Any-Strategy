import networkx as nx
import warnings
import math
import copy
import yaml

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
                 from_iros_ts: bool = False,
                 plot_auto_graph: bool = False,
                 plot_trans_graph: bool = False,
                 weighting: str = 'weightinglinear',
                 alpha: float = 1.0,
                 show_weight: bool = True,
                 observe_next_on_trans: bool = True,
                 complete_graph_players: List = ['adam', 'eve'],
                 integrate_accepting: bool = False) -> 'ProductAutomaton()':
        self._trans_sys: Optional[TwoPlayerGraph] = copy.deepcopy(trans_sys)
        self._auto_graph: Union[DFAGraph, PDFAGraph] = copy.deepcopy(automaton)

        self._absorbing = absorbing
        self._finite = finite
        self._from_iros_ts = from_iros_ts

        self._plot_auto_graph = plot_auto_graph
        self._plot_trans_graph = plot_trans_graph

        self._show_weight: bool = show_weight
        self._observe_next_on_trans = observe_next_on_trans
        self._complete_graph_players = complete_graph_players
        self._integrate_accepting = integrate_accepting

        self._multiple_weights = False

        if isinstance(automaton, PDFAGraph):
            self._multiple_weights = True
            self._weighting = self._choose_weighting(weighting, alpha)

        TwoPlayerGraph.__init__(self,
                                graph_name=graph_name,
                                config_yaml=config_yaml,
                                save_flag=save_flag)

    def compose_graph(self):

        # throw a warning if the DFA does NOT contain any symbol that appears in the set of observations in TS
        # if not self._check_ts_ltl_compatability():
        #     warnings.warn("Please make sure that the formula is composed of symbols that are part of the aps in the TS")

        self._extend_graphs()

        if self._absorbing:

            if self._from_iros_ts:
                self.construct_product_absorbing_from_iros_ts(finite=self._finite)

            else:
                self.construct_product_absorbing()

            # all the edges from the sys node to the absorbing state should have weight zero.
            if not isinstance(self._auto_graph, PDFAGraph):
                self._add_sys_to_abs_states_w_zero_wgt()

        else:
            self.construct_product()

        if self._integrate_accepting:
            self._integrate_accepting_states()

        self._sanity_check(debug=True)

    def _extend_graphs(self):

        # First, Extend PDFA graph to a single accepting state
        for _n in self._auto_graph.get_accepting_states():
            self._auto_graph._graph.nodes[_n]['accepting'] = False

            prob = self._auto_graph._graph.nodes[_n]['final_probability']
            transition_formula = f'(true)'
            transition_expr = parse_guard(transition_formula)
            self._auto_graph.add_edge(
                _n,
                AUTO_ACCEPTING_STATE_NAME,
                symbol='true',
                prob=prob,
                weight=float(-math.log(prob)),
                guard=transition_expr,
                guard_formula=transition_formula)

        self._auto_graph.add_state(AUTO_ACCEPTING_STATE_NAME,
                                   accepting=True,
                                   final_probability=1.0)

        if self._plot_auto_graph:
            orig_name = self._auto_graph._graph.name
            self._auto_graph._graph.name = orig_name + '_extended'
            self._auto_graph.plot_graph()

        # Second, Extend every node in TS to a single absorbing state
        # to align with the accepting state in PDFA
        for (_n, attr) in self._trans_sys.get_states():
            self._trans_sys.add_edge(_n, TS_EXTRA_STATE_NAME,
                                     weight=0,
                                     actions='NoAction',
                                     player=attr['player'])

        self._trans_sys.add_state(TS_EXTRA_STATE_NAME,
                                  ap='', player='eve')

        if self._plot_trans_graph:
            orig_name = self._trans_sys._graph.name
            self._trans_sys._graph.name = orig_name + '_extended'
            self._trans_sys.plot_graph()

    def _integrate_accepting_states(self):
        if not isinstance(self._auto_graph, PDFAGraph):
            return

        for _n in self.get_accepting_states():
            # Check if absorbing, then delete
            # TODO: What if the absorbing state is also an init state
            _preds = [_m for _m in self._graph.predecessors(_n)]

            self._graph.nodes[_n]['accepting'] = False
            self._graph.nodes[_n]['originalAccepting'] = True

            # Turn node's accepting to False
            # Add an edge between node to the accepting state
            self.add_edge(_n, PROD_ACCEPTING_STATE_NAME,
                        weight=0,
                        actions='NoAction',
                        player='eve')

        self.add_state(PROD_ACCEPTING_STATE_NAME,
                       ts=None, dfa=None, player='eve', accepting=True, ap=None)

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
        t_init = self._trans_sys.get_initial_states()[0]
        q_init = self._auto_graph.get_initial_states()[0]
        _n_init = self.composition(t_init, q_init)

        # Create Queues
        queue = Queue()
        visited = defaultdict(lambda: False)

        queue.push(_n_init)
        visited[_n_init] = True

        # Create transitions until there is no more
        while queue.is_empty():
            _u_prod_node = queue.pop()

            _u_ts_node = _u_prod_node[0]
            _u_a_node = _u_prod_node[1]

            # Next we check for all transitions in TS
            for _v_ts_node in self._trans_sys._graph.successors(_u_ts_node):

                # Get all info, cuz it's 100% sure we transition to each node in TS

                # Assumes only one ap exists in each node in TS
                ap = self._trans_sys._graph.nodes[_v_ts_node].get('ap')
                # Assumes only one edge exists betw. nodes in TS
                ts_action = self._trans_sys.get_edge_attributes(_u_ts_node, _v_ts_node, 'actions')
                _weight = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0].get('weight')

                added_trans = []
                skipped_trans = []

                # Check if the trans in TS satisfies any trans in the Automaton specification
                for _v_a_node in self._auto_graph._graph.successors(_u_a_node):

                    _v_prod_node = self.composition(_v_ts_node, _v_a_node)

                    # Assumes multiple symbols/actions exist in one transition
                    # (== multiple edges)
                    _, weights, auto_actions = self._get_edge_and_node_data(_u_ts_node,
                                                                            _v_ts_node,
                                                                            _u_a_node,
                                                                            _v_a_node)

                    # For each symbol, check if ap satisfies any of them
                    for weight, auto_action in zip(weights, auto_actions):
                        # determine if a transition is possible or not, if yes then add that edge
                        exists, _ = self._check_transition(_u_ts_node, _v_ts_node,
                                                           _u_a_node, _v_a_node,
                                                           _v_prod_node,
                                                           action=auto_action,
                                                           obs=ap)
                        if exists:
                            self._add_transition_absorbing(_u_prod_node,
                                                            _v_prod_node,
                                                            weight=weight,
                                                            action=ts_action)
                            # check if it's already been visited
                            if not visited[_v_prod_node]:
                                queue.push(_v_prod_node)
                                visited[_v_prod_node] == True
                                added_trans.append((_u_prod_node, _v_prod_node))
                        else:
                            skipped_trans.append((_u_prod_node, _v_prod_node, ts_action))

                # Case: Automaton doesn't include AP from TS
                # Keep adding transition, but with weight of inf (dead edge)
                if self._complete_graph and len(added_trans) == 0:
                    for skipped_tran in skipped_trans:
                        _u_ts_node = skipped_tran[0][0]
                        # Only create complete graph for adam nodes for cleanness
                        if self._trans_sys._graph.nodes[_u_ts_node].get('player') == 'adam':
                            self._add_transition_absorbing(skipped_tran[0],
                                                        skipped_tran[1],
                                                        weight=math.inf,
                                                        action=skipped_tran[2])

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
                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):
                        _v_prod_node = self._composition(_v_ts_node, _v_a_node)

                        # get relevant details that need to be added to the composed node and edge
                        ap, weights, auto_actions = self._get_edge_and_node_data(_u_ts_node,
                                                                               _v_ts_node,
                                                                               _u_a_node,
                                                                               _v_a_node)

                        ts_action = self.get_edge_attributes(_u_ts_node, _v_ts_node, 'actions')

                        for weight, auto_action in zip(weights, auto_actions):
                            # determine if a transition is possible or not, if yes then add that edge
                            exists, _v_prod_node = self._check_transition(_u_ts_node,
                                                                        _v_ts_node,
                                                                        _u_a_node,
                                                                        _v_a_node,
                                                                        _v_prod_node,
                                                                        action=auto_action,
                                                                        obs=ap)

                            if exists:
                                self._add_transition(_u_prod_node,
                                                    _v_prod_node,
                                                    weight=weight,
                                                    action=ts_action)

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

                    added_transition_record = defaultdict(lambda: [])
                    skipped_transition_record = defaultdict(lambda: [])

                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):

                        # if the next node is an absorbing state, then we don't compose v_ts_node and v_a_node
                        if _v_a_node in self._auto_graph.get_absorbing_states():
                            _v_prod_node = self._add_prod_state(_v_a_node, _v_a_node)
                        else:
                            _v_prod_node = self._composition(_v_ts_node, _v_a_node)

                        # get relevant details that need to be added to the composed node and edge
                        ap, weights, auto_actions = self._get_edge_and_node_data(_u_ts_node,
                                                                               _v_ts_node,
                                                                               _u_a_node,
                                                                               _v_a_node)

                        ts_action = self._trans_sys.get_edge_attributes(_u_ts_node, _v_ts_node, 'actions')

                        for weight, auto_action in zip(weights, auto_actions):
                            exists, _v_prod_node = self._check_transition_absorbing(_u_ts_node,
                                                                                    _v_ts_node,
                                                                                    _u_a_node,
                                                                                    _v_a_node,
                                                                                    _v_prod_node,
                                                                                    action=auto_action,
                                                                                    obs=ap)
                            if exists:
                                self._add_transition_absorbing(_u_prod_node,
                                                              _v_prod_node,
                                                              weight=weight,
                                                              action=ts_action)
                                added_transition_record[ap].append((_u_prod_node, _v_prod_node))
                            else:
                                skipped_transition_record[ap].append((_u_prod_node, _v_prod_node, ts_action))

                    # Case: Automaton doesn't include AP from TS
                    # Keep adding transition, but with weight of inf (dead edge)
                    if len(self._complete_graph_players)>0:
                        for ap, skipped_trans in skipped_transition_record.items():

                            added_trans = added_transition_record[ap]

                            if len(added_trans) == 0:
                                for skipped_tran in skipped_trans:
                                    _n = skipped_tran[0][0]
                                    # Only create complete graph for adam nodes for cleanness
                                    if self._trans_sys._graph.nodes[_n].get('player') in self._complete_graph_players:
                                        self._add_transition_absorbing(skipped_tran[0],
                                                                    skipped_tran[1],
                                                                    weight=math.inf,
                                                                    action=skipped_tran[2])

    def construct_product_absorbing_from_iros_ts(self, finite: bool):
        """
        A function that helps build the composition of TS and DFA where we compress the all
        absorbing nodes (A node that only has a transition to itself) in product automation.

        This method of building the product automaton aligns with the iros 17 abstraction in which we do not have
        a set of player but in fact we have a set of human and system actions from each state. To identify each action
        as human and system action, we add an attribute to that edge : player = "eve"\"adam".

        This method add this extra edge attribute by calling the _add_transition...._for_iros_ts() methods both in
        case finite or otherwise.

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
                    for _v_a_node in self._auto_graph._graph.successors(_u_a_node):

                        # if the next node is an absorbing state, then we don't compose v_ts_node and v_a_node
                        if _v_a_node in self._auto_graph.get_absorbing_states():
                            _v_prod_node = self._add_prod_state(_v_a_node, _v_a_node)
                        else:
                            _v_prod_node = self._composition(_v_ts_node, _v_a_node)

                        # get relevant details that need to be added to the composed node and edge
                        ap, weight, auto_action = self._get_edge_and_node_data(_u_ts_node,
                                                                               _v_ts_node,
                                                                               _u_a_node,
                                                                               _v_a_node)

                        ts_player = self._trans_sys.get_edge_attributes(_u_ts_node, _v_ts_node, "player")
                        ts_action = self._trans_sys.get_edge_attributes(_u_ts_node, _v_ts_node, 'actions')

                        exists, _v_prod_node = self._check_transition_absorbing(_u_ts_node,
                                                                                _v_ts_node,
                                                                                _u_a_node,
                                                                                _v_a_node,
                                                                                _v_prod_node,
                                                                                action=auto_action,
                                                                                obs=ap)
                        if exists:
                            if finite:
                                self._add_transition_absorbing_finite_for_iros_ts(_u_prod_node,
                                                                                  _v_prod_node,
                                                                                  weight=weight,
                                                                                  max_weight=max_w,
                                                                                  action=ts_action,
                                                                                  edge_player=ts_player)
                            else:
                                self._add_transition_absorbing_for_iros_ts(_u_prod_node,
                                                                           _v_prod_node,
                                                                           weight=weight,
                                                                           max_weight=max_w,
                                                                           action=ts_action,
                                                                           edge_player=ts_player)

    def _add_transition(self, _u_prod_node, _v_prod_node, weight: float, action: str):
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
                          actions=action)

    def _add_transition_absorbing_finite_for_iros_ts(self,
                                                     _u_prod_node,
                                                     _v_prod_node,
                                                     weight: float,
                                                     max_weight: float,
                                                     action: str,
                                                     edge_player: str,
                                                     accepting_edge_player: str = "eve",
                                                     trap_edge_player: str = "eve") -> None:
        """
        A helper method with the same functionality as _add_transition_absorbing_finite() method. . In this method
        we add edge player attribute to each edge on the graph on the product graph.

        For the self loop of the accepting state as well as the trap state the default value of player is eve.

        :param _u_prod_node:
        :param _v_prod_node:
        :return: Updated the graph with the edge
        """
        _valid_player_list = ["eve", "adam"]

        if edge_player not in _valid_player_list:
            warnings.warn(f"Opps looks like the edge {_u_prod_node} ----> {_v_prod_node} has player {edge_player} in TS"
                          f" which is not a valid type. ")

        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            if _u_prod_node in self._auto_graph.get_absorbing_states():
                # accepting state
                if self._auto_graph._graph.nodes[_u_prod_node].get('accepting'):
                    self.add_edge(_u_prod_node, _v_prod_node,
                                  weight=0,
                                  actions=action,
                                  player=accepting_edge_player)
                # trap state
                else:
                    self.add_edge(_u_prod_node, _v_prod_node,
                                  weight=(-1 * math.inf),
                                  actions=action,
                                  player=trap_edge_player)
            else:
                self.add_edge(_u_prod_node, _v_prod_node,
                              weight=weight,
                              actions=action,
                              player=edge_player)

    def _add_transition_absorbing_for_iros_ts(self,
                                              _u_prod_node: tuple,
                                              _v_prod_node: tuple,
                                              weight: float,
                                              max_weight: float,
                                              action: str,
                                              edge_player: str,
                                              accepting_edge_player: str = "eve",
                                              trap_edge_player: str = "eve") -> None:
        """
        A helper method similar to _add_transition_absorbing() method. In this method we add edge player attribute
        to each edge on the graph on the product graph.

        For the self loop of the accepting state as well as the trap state the default value of player is eve.

        :param _u_prod_node:
        :param _v_prod_node:
        :return: Updated graph with the edge
        """

        _valid_player_list = ["eve", "adam"]

        if edge_player not in _valid_player_list:
            warnings.warn(f"Ops looks like the edge {_u_prod_node} ----> {_v_prod_node} has player {edge_player} in TS"
                          f" which is not a valid type.")

        if not self._graph.has_edge(_u_prod_node, _v_prod_node):
            if _u_prod_node in self._auto_graph.get_absorbing_states():
                # accepting state
                if self._auto_graph._graph.nodes[_u_prod_node].get('accepting'):
                    self.add_edge(_u_prod_node, _v_prod_node,
                                  weight=0,
                                  actions=action,
                                  player=accepting_edge_player)
                # trap state
                else:
                    self.add_edge(_u_prod_node, _v_prod_node,
                                  weight=max_weight,
                                  actions=action,
                                  player=trap_edge_player)
            else:
                self.add_edge(_u_prod_node, _v_prod_node,
                              weight=weight,
                              actions=action,
                              player=edge_player)

    def _add_transition_absorbing(self,
                                  _u_prod_node,
                                  _v_prod_node,
                                  weight: float,
                                  action: str) -> None:
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
                              actions=action)
            else:
                self.add_edge(_u_prod_node, _v_prod_node,
                              weight=weight,
                              actions=action)

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

            # if the next node belongs to an absorbing state
            if _v_a_node in self._auto_graph.get_absorbing_states():
                _v_prod_node = self._add_prod_state(_v_a_node, _v_a_node)

            else:
                _v_prod_node = self._composition(_v_ts_node, _v_a_node)

            # TODO: Ask Karan why this statement is always True.
            # return True, _v_prod_node
            if action.formula == "(true)" or action.formula == "1":
                return True, _v_prod_node
            else:
                return action.check(obs), _v_prod_node

        else:
            warnings.warn(f"Looks like the node {_u_ts_node} in graph {self._trans_sys._graph_name} does"
                          f" not have a valid player assigned to it")

    def _get_edge_and_node_data(self, _u_ts_node, _v_ts_node, _u_auto_node, _v_auto_node) -> Tuple[str, float, str]:
        """
        A helper method that returns

            1. observation at the current node in TS (i.e at -observation at _u_ts_node)
            2. weight : The edges weight from the TS between _u_ts_node and _v_ts_node
            3. transition label : The transition label from the automation between _u_auto_node and _v_auto_node
        :param _u_ts_node: The current node in TS
        :param _v_ts_node:  Neighbour/successor of the current node in TS
        :param _u_auto_node: The current node in Automation - DFA
        :param _v_auto_node: Neighbour/successor of the current node in DFA
        :return: A tuple of observation, weight, and transition label
        """

        if self._observe_next_on_trans:
            _observation = self._trans_sys._graph.nodes[_v_ts_node].get('ap')
        else:
            _observation = self._trans_sys._graph.nodes[_u_ts_node].get('ap')

        try:
            _weight = self._trans_sys._graph.get_edge_data(_u_ts_node, _v_ts_node)[0].get('weight')
        except:
            warnings.warn(f"The edge from {_u_ts_node} to {_v_ts_node} does not contain the attribute 'weight'."
                          f"Setting the edge weight to 0 while constructing the product")
            _weight = 0

        auto_edge_data = self._auto_graph._graph.get_edge_data(_u_auto_node, _v_auto_node)
        num_auto_edges = len(auto_edge_data)

        _weights = []
        _automaton_labels = []

        for i_edge in range(num_auto_edges):
            if self._multiple_weights:
                try:
                    _a_weight = auto_edge_data[i_edge].get('weight')
                    _weight = self._weighting(_weight, _a_weight)
                except:
                    msg = f"The weight from edge {_u_auto_node} to {_v_auto_node} does not exist"
                    warnings.warn(msg)
                    _weight = 0

            _automaton_label = auto_edge_data[i_edge]['guard']

            _weights.append(_weight)
            _automaton_labels.append(_automaton_label)

        return _observation, _weights, _automaton_labels

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
                # self.add_accepting_state(_p_node)

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
        # check is the graph is total or not by looping through every node and add a self-loop of weight max(W)
        # to every node that does not  have a successor
        max_w: float = self._trans_sys.get_max_weight()
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 0:
                if debug:
                    print("=====================================")
                    print(f"Adding self loop of weight - {max_w} to the node {_n}")
                    print("=====================================")
                self._graph.add_weighted_edges_from([(_n,
                                                      _n, 0)])
                                                      # math.inf)])
                # -1 * max_w)])

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

    def _choose_weighting(self, weighting_name: str, alpha: float):
        """
        Choose weighting method

        :param weighting_name:
        :param alpha:
        :return:    A function with two inputs w1, w2
        """
        if weighting_name == 'linear':
            return lambda w1, w2: w1 + w2
        elif weighting_name == 'weightedlinear':
            return lambda w1, w2: w1 + alpha * w2
        elif weighting_name == 'automatonOnly':
            return lambda w1, w2: w2
        else:
            raise ValueError(f'No such weighting as {weighting_name}')

    def fancy_graph(self, color=("lightgrey", "red", "purple", "cyan")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            ap = n[1].get('ap')
            val = n[1].get('val')
            val = 'None' if val is None else f'{val:.2f}'
            xlabel = f"ap:{ap}, val:{val}"
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
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "circle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = str(edge[2].get('actions'))

            if self._show_weight:
                weight = edge[2].get('weight')
                label += f': {weight:.2f}'

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
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)


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

        trans_sys = kwargs['trans_sys'] if 'trans_sys' in kwargs else None
        automaton = kwargs['automaton'] if 'automaton' in kwargs else None

        if trans_sys is not None and automaton is not None:
            self._instance.construct_graph()
            self._instance.compose_graph()

        if prune:
            self._instance.prune_graph(debug=debug)

        if plot:
            self._instance.plot_graph()

        return self._instance
