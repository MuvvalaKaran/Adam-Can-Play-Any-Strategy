import warnings
import math
import operator
import copy

from typing import Optional, Dict, List, Tuple

# import local packages
from src.graph import TwoPlayerGraph


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

    def synthesize_str(self):

        _accepting_states = self.game.get_accepting_states()
        _start_state = self.game.get_initial_states()[0][0]

        for _acc_s in _accepting_states:
            self.str_map[_acc_s]["cost"] = 0

        # a flag to terminate when all the states in G have converged to their respective costs.
        _converged = False

        while not _converged and self.str_map[_start_state]["cost"] > self.energy_bound:
            __prev_str_map = copy.deepcopy(self.str_map)
            _converged = True

            for _s in self.game._graph.nodes():
                # _env_cost = max([self.str_map[_succ_s]
                #                  for _succ_s in self.game._graph.successors(_s)
                #                  if self.game._graph.edges[_s, _succ_s, 0].get("player") == "adam"])
                _env_cost = self._get_max_env_cost_from_s(_s)
                _sys_cost_edge, _sys_cost = self._get_min_sys_cost_from_s(_s)

                self.str_map[_s]["cost"] = max(_env_cost, _sys_cost)

                self.str_map[_s]["action"] = self.game.get_edge_attributes(_sys_cost_edge[0],
                                                                           _sys_cost_edge[1],
                                                                           "actions")

                for _succ_s in self.game._graph.successors(_s):
                    if __prev_str_map[_succ_s] != self.str_map[_succ_s]:
                        _converged = False

    def _get_max_env_cost_from_s(self, state: tuple) -> int:
        """
        Find the env transition to the state with the highest cost value
        :param state:
        :return:
        """
        _succ_s_costs: List[int] = []

        for _succ_s in self.game._graph.successors(state):
            if self.game._graph.edges[state, _succ_s, 0].get("player") == "adam":
                _succ_s_costs.append(self.str_map[_succ_s]["cost"])

        return max(_succ_s_costs)

    def _get_min_sys_cost_from_s(self, state: tuple) -> Tuple[tuple, int]:
        """
        Find the sys transition to the state with the least cost value
        :param state:
        :return:
        """

        _succ_s_costs: List[Tuple[tuple, int]] = []

        for _succ_s in self.game._graph.successors(state):
            if self.game._graph.edges[state, _succ_s, 0].get("player") == "eve":
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