import warnings
import math
import operator
import sys
import copy

from typing import Optional, Union, Dict, List, Tuple
from collections import deque

# import local packages
from src.graph import TwoPlayerGraph
from src.payoff import Payoff


class CumulativePayoff:

    def __init__(self,
                 game: TwoPlayerGraph):
        self._game = game
        self._str_map: Optional[Dict[Dict[str, int]]] = {}
        self._initialize_str()
        self._convert_weights_to_positive_costs(plot=False)

    @property
    def game(self):
        return self._game

    @property
    def str_map(self):
        return self._str_map

    @game.setter
    def game(self, game: TwoPlayerGraph):
        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = game

    def solve(self, debug: bool = False, plot: bool = False):

        _iter_count: int = 1
        _start_state = self.game.get_initial_states()[0][0]

        while self.iterate():

            if debug:
                print(f"The cost from the initial state {_start_state} after {_iter_count}"
                      f" iterations is {abs(self.str_map[_start_state]['cost'])}")

            _iter_count += 1

        # add state cost as xlabels and plot it.
        if plot:
            self._plot_game_with_state_costs(only_eve=False)

    def iterate(self):
        _accepting_states = self.game.get_accepting_states()
        _trap_state = self.game.get_trap_states()
        _absorbing_states = _accepting_states + _trap_state

        _state = self.game.get_initial_states()[0][0]

        for _acc_s in _accepting_states:
            self.str_map[_acc_s]["cost"] = 0

        # queue to keep track of nodes visited
        _node_visited = deque()
        _node_queue = deque()

        # a flag to terminate when all the states in G have converged to their respective costs.
        _progress = False

        _node_queue.append(_state)

        # assuming this is a cooperative game
        while _node_queue:
            _state = _node_queue.popleft()

            if _state not in _node_visited and _state not in _absorbing_states:
                _node_visited.append(_state)

                _curr_cost = self.str_map[_state]["cost"]

                # compute the best sys action
                if self.game.get_state_w_attribute(_state, "player") == "eve":
                    _best_edge, _best_cost = self._get_min_sys_cost_from_s(_state)
                elif self.game.get_state_w_attribute(_state, "player") == "adam":
                    _best_edge, _best_cost = self._get_min_env_cost_from_s(_state)
                else:
                    warnings.warn(f"Please make sure that ever state in graph {self.game._graph_name}. "
                                  f"Currently the state {_state} does not have a valid player!")
                    sys.exit(-1)

                if _best_cost <= _curr_cost:
                    self.str_map[_state]["cost"] = _best_cost
                    self.str_map[_state]["action"] = _best_edge[1]
                if _best_cost < _curr_cost:
                    _progress = True

                for _next_n in self.game._graph.successors(_state):
                    _node_queue.append(_next_n)

        return _progress

    def _get_min_sys_cost_from_s(self, state: tuple) -> Tuple[tuple, int]:
        """
        Find the sys transition to the next human state with the least cost associated with that state
        :param state:
        :return:
        """

        _succ_s_costs: List[Tuple[tuple, int]] = []

        for _succ_s in self.game._graph.successors(state):
            if state != _succ_s:
                val = self.game.get_edge_weight(state, _succ_s) + self.str_map[_succ_s]["cost"]
                _succ_s_costs.append(((state, _succ_s), val))

        return min(_succ_s_costs, key=operator.itemgetter(1))

    def _get_min_env_cost_from_s(self, state: tuple) -> Tuple[tuple, int]:
        """
        Find the env transition to the next sys state with the least cost associated with that state
        :param state:
        :return:
        """

        _succ_s_costs: List[Tuple[tuple, int]] = []
        # val is equivalent to the cost of the next state because there is no weight associated with edges that
        # originate from human states

        for _succ_s in self.game._graph.successors(state):
            if state != _succ_s:
                val = self.str_map[_succ_s]["cost"]
                _succ_s_costs.append(((state, _succ_s), val))

        return min(_succ_s_costs, key=operator.itemgetter(1))

    def _convert_weights_to_positive_costs(self, plot: bool = False):
        """
        A helper method that converts the -ve weight that represent cost to positive edge weights for a given game.
        :return:
        """

        for _e in self.game._graph.edges.data("weight"):
            _u = _e[0]
            _v = _e[1]

            _curr_weight = _e[2]
            if _curr_weight < 0:
                _new_weight: Union[int, float] = -1 * _curr_weight
            else:
                _new_weight: Union[int, float] = _curr_weight

            self.game._graph[_u][_v][0]["weight"] = _new_weight

        if plot:
            self.game.plot_graph()

    def _plot_game_with_state_costs(self, only_eve: bool = False):
        """
        A method to help visualize the strategy we just computed and the cost associated with each state.
        :return:
        """
        # add strategy attribute to a copy of the graph
        _graph = copy.deepcopy(self.game)

        _graph.set_edge_attribute('strategy', False)

        # add state costs as xlabels
        self._add_xlabels_to_graph(_graph)

        if only_eve:
            self._add_strategy_flag_only_eve(_graph)
        else:
            self._add_strategy_flag(_graph)

        _graph.plot_graph()

    def _add_xlabels_to_graph(self, graph: TwoPlayerGraph):
        """
        A method to replace all the original xlabels (if any) with the state cost computed by the solve() method
        :param graph: A local copy of the graph
        :return:
        """

        for _node in self.str_map.keys():
            graph.add_state_attribute(_node, "ap", self.str_map[_node]["cost"])

    def get_str_dict(self) -> Dict[tuple, str]:
        """
        As str_map is a dictionary that also has the cost associated with it, we make use of this method to return a
        strategy dictionary which is a mapping of the best SYSTEM ACTION from each state in the game.
        :return:
        """
        if len(self.str_map.keys()) == 0:
            warnings.warn("Please make sure that you solve the game before you access the strategy."
                          "Use the solve() method to compute the costs and the respective actions to take")

        str_dict = {}
        _accepting_states = self.game.get_accepting_states()
        _trap_state = self.game.get_trap_states()
        _absorbing_states = _accepting_states + _trap_state

        for _state, _state_map in self.str_map.items():
            if _state_map["action"] is not None:
                str_dict.update({_state: _state_map["action"]})

            # only absorbing state don't have action in the strategy map as they are skipped over in the solve() method
            else:
                if _state in _absorbing_states:
                    str_dict.update({_state: _state})

        return str_dict

    def _initialize_str(self):

        # map is a dictionary that maps each state to a action and cost value
        for _s in self.game._graph.nodes():
            self.str_map.update({_s: {"action": None, "cost": math.inf}})

    def _add_strategy_flag_only_eve(self, graph: TwoPlayerGraph):
        """
        A helper method that adds a strategy attribute to the edges of the game that belong to the strategy dict
        computed using the solve() method.

        Effect: loops over the dict and updates attribute to True if that node exists in the strategy dict and
        belongs to eve ONLY.

        :param graph: A local copy of the graph
        :return:
        """
        str_dict = self.get_str_dict()
        for curr_node, next_node in str_dict.items():
                if graph._graph.nodes[curr_node].get("player") == "eve":
                    graph._graph.edges[curr_node, next_node, 0]['strategy'] = True

    def _add_strategy_flag(self, graph: TwoPlayerGraph):
        """
        A helper method that adds a strategy attribute to the edges of the game that belong to the strategy dict
         computed using the solve method().

        Effect: loops over the dict and updates attribute to True if that node exists in the strategy dict.
        :param graph: A local copy of the graph
        :return:
        """
        str_dict = self.get_str_dict()
        for curr_node, next_node in str_dict.items():
            graph._graph.edges[curr_node, next_node, 0]['strategy'] = True


