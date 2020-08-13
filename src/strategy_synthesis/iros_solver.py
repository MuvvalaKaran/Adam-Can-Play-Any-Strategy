import warnings
import math
import operator
import random
import sys

import numpy as np

from numpy import ndarray
from typing import Optional, Dict, List, Tuple, Union
from collections import deque

# import local packages
from src.graph import TwoPlayerGraph


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
        if not isinstance(bound_value, int) or bound_value < 0:
            warnings.warn("Please ensure that the bound on the energy is an integer and is positive definite(>=0)")

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

    def get_str_dict(self) -> Dict[tuple, str]:
        """
        As str_map is a dictionary that also has the cost associated with it, we make use of this method to return a
        strategy dictionary which is a mapping of the best SYSTEM ACTION from each state in the game.
        :return:
        """
        if len(self.str_map.keys()) == 0:
            warnings.warn("Please make sure that you solve the game before you access the strategy."
                          "Use the solve() method to compute the bounded winning strategy/")

        str_dict = {}

        for _state, _state_map in self.str_map.items():
            str_dict.update({_state: _state_map["action"]})

        return str_dict

    def _get_next_state_w_edge_label(self, _curr_state: tuple, _edge_label: str) -> Optional[tuple]:
        """
        A helper method to return the next state given the current state and a valid edge label
        :param _curr_state:
        :param _edge_label:
        :return:
        """
        _next_node = None
        _neigh_states = [_next_n for _next_n in self.game._graph.successors(_curr_state)]

        # check the edge label between the current node and the neighbouring node
        for _n in _neigh_states:
            if _edge_label == self.game.get_edge_attributes(_curr_state, _n, "actions"):
                _next_node = _n

        return _next_node

    def _get_rand_state_human_action(self, _curr_state: tuple):
        """
        A helper method that randomly picks a human edge from the given state and returns the corresponding next state.
        :param _curr_state: The current node in the game
        :return:
        """

        _adams_choices = []

        for _next_n in self.game._graph.successors(_curr_state):
            if self.game.get_edge_attributes(_curr_state, _next_n, "player") == "adam" and _next_n[0]:
                _adams_choices.append(_next_n)

        return random.choice(_adams_choices)

    def _epsilon_greedy_choose_action(self,
                                      _state: tuple,
                                      str_dict: Dict[tuple, str],
                                      epsilon: float,
                                      _absoring_states: List[tuple],
                                      human_can_intervene: bool = False) -> Tuple[tuple, bool]:
        """
        Choose an action according to epsilon greedy algorithm

        Using this policy we either select a system action with epsilon probability and the human can select the
        optimal action (as given in the str dict) with 1-epsilon probability.

        This method returns an action from a state based on the above algorithm.
        :param _state: The current state in the game from which we need to pick an action.
        :return: Tuple(next_state, flag) . flag = True if human decides to take an action.
        """
        _did_human_move = False
        _human_moves_remaining = _state[0][1]

        if human_can_intervene:
            # rand() return a floating point number between [0, 1)
            if np.random.rand() < epsilon or _human_moves_remaining == 0:
                # pick the best state that the system wants to transit to
                best_sys_action = str_dict[_state]
                _next_state: tuple = self._get_next_state_w_edge_label(_state, best_sys_action)

            else:
                # pick a random human action
                _next_state: tuple = self._get_rand_state_human_action(_state)

            if _next_state is None:
                warnings.warn(f"Could not find a state to transit to from {_state}")
                sys.exit(-1)

            if _next_state not in _absoring_states:
                if _next_state[0][1] != _state[0][1]:
                    _did_human_move = True
        else:
            best_sys_action = str_dict[_state]
            _next_state: tuple = self._get_next_state_w_edge_label(_state, best_sys_action)

        return _next_state, _did_human_move

    def get_controls_from_str_minigrid(self,
                                       epsilon: float,
                                       debug: bool = False,
                                       max_human_interventions: int = 1,) -> List[Tuple[str, ndarray, int]]:
        """
        As this game DOES NOT have explicit human and env nodes, we have distinct human and system action from each
        state

        This method uses epsilon-greedy method to choose an action and return the sequence of positions
        :return: A sequence of labels that to be executed by the robot
        """
        if not isinstance(epsilon, float):
            try:
                epsilon = float(epsilon)
                if epsilon > 1:
                    warnings.warn("Please make sure that the value of epsilon is <=1")
            except ValueError:
                print(ValueError)

        if not isinstance(max_human_interventions, int) or max_human_interventions < 0:
            warnings.warn("Please make sure that the max human intervention bound should >= 0")

        # compute the str_dict and str_map also cost associated with it
        str_dict = self.get_str_dict()

        _start_state = self.game.get_initial_states()[0][0]
        _total_human_intervention = _start_state[0][1]
        _accepting_states = self.game.get_accepting_states()
        _trap_states = self.game.get_trap_states()

        _absorbing_states = _accepting_states + _trap_states
        _human_interventions: int = 0
        _visited_states = []
        _position_sequence = []

        curr_sys_node = _start_state

        _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
        next_sys_node = self._epsilon_greedy_choose_action(curr_sys_node,
                                                           str_dict,
                                                           epsilon=epsilon,
                                                           _absoring_states=_absorbing_states,
                                                           human_can_intervene=_can_human_intervene)[0]

        (x, y) = next_sys_node[0][0]
        _human_interventions: int = _total_human_intervention - next_sys_node[0][1]

        next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)

        _visited_states.append(curr_sys_node)
        _visited_states.append(next_sys_node)

        _entered_absorbing_state = False

        while 1:
            _position_sequence.append(next_pos)

            curr_sys_node = next_sys_node

            # update the next sys node only if you not transiting to an absorbing state
            if curr_sys_node not in _absorbing_states:
                _can_human_intervene: bool = True if _human_interventions < max_human_interventions else False
                next_sys_node = self._epsilon_greedy_choose_action(curr_sys_node,
                                                                   str_dict,
                                                                   epsilon=epsilon,
                                                                   _absoring_states=_absorbing_states,
                                                                   human_can_intervene=_can_human_intervene)[0]

            # if transiting to an absorbing state then due to the self transition the next sys node will be the same as
            # the current env node which is an absorbing itself. Technically an absorbing state IS NOT assigned any
            # player.
            else:
                next_sys_node = curr_sys_node

            if next_sys_node in _visited_states:
                break

            # if you enter a trap/ accepting state then do not add that transition in _pos_sequence
            elif next_sys_node in _absorbing_states:
                _entered_absorbing_state = True
                break
            else:
                x, y = next_sys_node[0][0]
                _human_interventions: int = _total_human_intervention - next_sys_node[0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(next_sys_node)

        if not _entered_absorbing_state:
            _position_sequence.append(next_pos)

        if debug:
            print([_n for _n in _position_sequence])

        return _position_sequence