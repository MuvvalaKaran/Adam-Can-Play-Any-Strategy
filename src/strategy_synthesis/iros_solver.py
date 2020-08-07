import warnings
import math
import operator
import copy

from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# import local packages
from src.graph import TwoPlayerGraph
from helper_methods import deprecated

class IrosStrategySynthesis:

    def __init__(self, game: TwoPlayerGraph, energy_bound: int = 5,  debug: bool = False):
        self._game = game
        self.__energy_bound: int = energy_bound
        self._str_map: Optional[Dict[str, Dict[str, int]]] = {}
        self._initialize_str()

    @property
    def game(self):
        return self._game

    @property
    def energy_bound(self):
        return self.__energy_bound

    @property
    def str_map(self):
        return self._str_map

    @game.setter
    def game(self, game: TwoPlayerGraph):
        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = game

    @energy_bound.setter
    def energy_bound(self, bound_value: int):
        if not isinstance(bound_value, int):
            warnings.warn("Please ensure that the bound on the energy is an integer")

        self.__energy_bound = bound_value

    def solve(self):
        _solved: bool = False
        _start_state = self.game.get_initial_states()[0][0]

        while (not _solved) and self.iterate():
            _solved = (abs(self.str_map[_start_state]["cost"]) <= self.energy_bound)

        _solved = (abs(self.str_map[_start_state]["cost"]) <= self.energy_bound)

        return _solved

    def iterate(self):

        _accepting_states = self.game.get_accepting_states()
        _trap_state = self.game.get_trap_states()
        _n = self.game.get_initial_states()[0][0]

        for _acc_s in _accepting_states:
            self.str_map[_acc_s]["cost"] = 0

        # dictionary to keep track of nodes visited
        _node_visited = defaultdict(lambda: False)

        # a flag to terminate when all the states in G have converged to their respective costs.
        _progress = False

        _progress = self._iterative_loop(_n, _progress, _node_visited)

        return _progress
        # for _next_n in self.game._graph.successors(_n):
        #     if _next_n != _n:
        #         self._iterative_loop(_next_n, _progress)

        # while not _converged and self.str_map[_start_state]["cost"] > self.energy_bound:
        # for _s in self.game._graph.nodes():
        # while _n:
        #     if _n not in _accepting_states + _trap_state:
        #
        #         # compute the best system action
        #         _sys_cost_edge, _sys_cost = self._get_min_sys_cost_from_s(_n)
        #
        #         # compute the worst env action
        #         if _n[0][1] != 0:
        #             _env_cost = self._get_max_env_cost_from_s(_n)
        #
        #             # get the max cost
        #             best_cost: int = max(_env_cost, _sys_cost)
        #         else:
        #             best_cost: int = _sys_cost
        #
        #         if best_cost <= self.str_map[_n]["cost"]:
        #             self.str_map[_n]["cost"] = best_cost
        #             self.str_map[_n]["action"] = self.game.get_edge_attributes(_sys_cost_edge[0],
        #                                                                        _sys_cost_edge[1],
        #                                                                        "actions")
        #
        #         if best_cost < self.str_map[_n]["cost"]:
        #             _progress = True

            # get one of the successors and break
            # _next_n = self.game._graph.successors(_n)
            # for _next_n in self.game._graph.successors(_n):
            #     _n = _next_n
            #     break

    def _iterative_loop(self, _state, _progress: bool, _node_visited: dict):
        _accepting_states = self.game.get_accepting_states()
        _trap_state = self.game.get_trap_states()
        while (_state not in _accepting_states) and (_state not in _trap_state):
            if not _node_visited[_state]:
                _node_visited[_state] = True
                # if _state not in _accepting_states + _trap_state:

                _curr_cost = self.str_map[_state]["cost"]

                # compute the best system action
                _sys_cost_edge, _sys_cost = self._get_min_sys_cost_from_s(_state)

                # compute the worst env action
                if _state[0][1] != 0:
                    _env_cost = self._get_max_env_cost_from_s(_state)

                    if _env_cost is not None:
                        # get the max cost
                        best_cost: int = max(_env_cost, _sys_cost)
                    else:
                        best_cost: int = _sys_cost

                else:
                    best_cost: int = _sys_cost

                if best_cost <= _curr_cost:
                    self.str_map[_state]["cost"] = best_cost
                    self.str_map[_state]["action"] = self.game.get_edge_attributes(_sys_cost_edge[0],
                                                                                   _sys_cost_edge[1],
                                                                                   "actions")

                if best_cost < _curr_cost:
                    _progress = True

                # get one of the successors and break
                # _next_n = self.game._graph.successors(_state)
                for _next_n in self.game._graph.successors(_state):
                    # if _next_n != _state:
                    _progress = self._iterative_loop(_next_n, _progress, _node_visited)
            else:
                return _progress

        return _progress

    def _get_max_env_cost_from_s(self, state: tuple) -> Optional[int]:
        """
        Find the env transition to the state with the highest cost value
        :param state:
        :return:
        """
        _succ_s_costs: List[int] = []

        for _succ_s in self.game._graph.successors(state):
            if self.game._graph.edges[state, _succ_s, 0].get("player") == "adam":
                _succ_s_costs.append(self.str_map[_succ_s]["cost"])

        if len(_succ_s_costs) == 0:
            return None
        return max(_succ_s_costs)

    def _get_max_sys_cost_from_s(self, state: tuple) -> Tuple[tuple, int]:
        """
        Find the sys transition to the state with the wost cost value
        :param state:
        :return:
        """

        _succ_s_costs: List[Tuple[tuple, int]] = []

        for _succ_s in self.game._graph.successors(state):
            if self.game._graph.edges[state, _succ_s, 0].get("player") == "eve":
                val = self.game.get_edge_weight(state, _succ_s) + self.str_map[_succ_s]["cost"]
                _succ_s_costs.append(((state, _succ_s), val))

        return max(_succ_s_costs, key=operator.itemgetter(1))

    def _get_min_sys_cost_from_s(self, state: tuple) -> Tuple[tuple, int]:
        """
        Find the sys transition to the state with the least cost value
        :param state:
        :return:
        """

        _succ_s_costs: List[Tuple[tuple, int]] = []

        for _succ_s in self.game._graph.successors(state):
            if self.game._graph.edges[state, _succ_s, 0].get("player") == "eve" and (not state == _succ_s) :
                val = self.game.get_edge_weight(state, _succ_s) + self.str_map[_succ_s]["cost"]
                _succ_s_costs.append(((state, _succ_s), val))

        return min(_succ_s_costs, key=operator.itemgetter(1))

    def _initialize_str(self):

        # map is a dictionary that maps each state to a action and cost value
        for _s in self.game._graph.nodes():
            self.str_map.update({_s: {"action": None, "cost": math.inf}})

    def print_map_dict(self):

        for _s in self.str_map.items():
            print(f"From state {_s} the strategy is to take action {self.str_map[_s]['action']} with"
                  f" cost {self.str_map[_s]['cost']}")