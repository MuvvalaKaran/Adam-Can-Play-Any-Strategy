import warnings
import math
import operator

from typing import Optional, Dict, List, Tuple, Union
from collections import deque

# import local packages
from ..graph import TwoPlayerGraph


class IrosStrategySynthesis:

    def __init__(self,
                 game: TwoPlayerGraph,
                 energy_bound: int = 5,
                 plot_modified_game: bool = False):
        self._game = game
        self.__energy_bound: int = energy_bound
        self._str_map: Optional[Dict[Dict[str, int]]] = {}
        self._initialize_str()
        self._convert_weights_to_positive_costs(plot=plot_modified_game)

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

    def solve(self, debug: bool = False):
        _solved: bool = False
        _start_state = self.game.get_initial_states()[0][0]
        _iter_count = 1

        while (not _solved) and self.iterate():
            _solved = (abs(self.str_map[_start_state]["cost"]) <= self.energy_bound)

            if debug:
                print(f"The cost from the initial state {_start_state} after {_iter_count}"
                      f" iterations is {abs(self.str_map[_start_state]['cost'])}")

            _iter_count += 1

        _solved = (abs(self.str_map[_start_state]["cost"]) <= self.energy_bound)

        return _solved

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

        while _node_queue:
            _state = _node_queue.popleft()

            if _state not in _node_visited and _state not in _absorbing_states:
                _node_visited.append(_state)

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

                for _next_n in self.game._graph.successors(_state):
                    _node_queue.append(_next_n)

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

    def _initialize_str(self):

        # map is a dictionary that maps each state to a action and cost value
        for _s in self.game._graph.nodes():
            self.str_map.update({_s: {"action": None, "cost": math.inf}})

    def print_map_dict(self):

        for _s, _v in self.str_map.items():
            print(f"From state {_s} the strategy is to take action {_v['action']} with"
                  f" cost {_v['cost']}")