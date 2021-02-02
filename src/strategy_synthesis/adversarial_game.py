import warnings
import numpy as np
import random

from numpy import ndarray
from typing import Optional, Iterable, Dict, List, Tuple
from collections import defaultdict
from collections import deque

# import local packages
from ...src.graph import graph_factory
from ...src.graph import TwoPlayerGraph


class ReachabilityGame:
    """
    A class that implements a Reachability algorithm to compute a set of winning region and the
    corresponding winning strategy for both the players in a Game G = (S, E). A winning strategy induces a play for the system player that
    satisfies the winning condition i.e or reach the accepting states on a given game.

    S : Set of all states in a given Game S = (S1 U S2) and (S1 ∩ S2 = nullset)
    W1 : Is the winning region for the system player from which it can force a visit to the accepting states
    W2 : Set compliment of W1 (S\W1) is the winning region for the evn player

    For a Reachability game the goal in terms of LTL could be written as : F(Accn States) i.e Eventually
    reach the accepting region. A strategy for the sys ensures that it can force a visit to the accn states from W1
    while the strategy for the env player is to remain in W2 or the trap region.

    Graph : The graph G is a two player game played between the system and the env player assuming env to be fully
    adversarial. G should be total and every node in G should be assigned a player
    """
    def __init__(self, game: TwoPlayerGraph, debug: bool = False):
        self._game = game
        self._sys_winning_region: Optional[Iterable] = None
        self._env_winning_region: Optional[Iterable] = None
        self._sys_str: Optional[dict] = None
        self._env_str: Optional[dict] = None

        self._sanity_check_player()
        self._sanity_check_total(debug=debug)

    @property
    def game(self):
        return self._game

    @property
    def sys_winning_region(self):
        return self._sys_winning_region

    @property
    def env_winning_region(self):
        return self._env_winning_region

    @property
    def sys_str(self):
        return self._sys_str

    @property
    def env_str(self):
        return self._env_str

    @game.setter
    def game(self, game: TwoPlayerGraph):

        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = game

    def _compute_no_of_node_successors(self) -> dict:
        """
        A helper method that initially compute the number of outgoing edges from a node.

        This variable is used to keep track of number of edges visited for a given node that belongs
        to the env player.

        :return:
        """
        num_out_edges = defaultdict(int)

        for _n in self.game._graph.nodes():
            num_out_edges[_n] = len(list(self.game._graph.successors(_n)))

        return num_out_edges

    def reachability_solver(self):
        """
        Implements the reachability solver by creating sub games

        If a env node i.e a node that has player == "adam" attribute and if that node belong's to the system's winning
        region, we do not add a strategy to it.
        :return:
        """

        num_out_edges = self._compute_no_of_node_successors()

        queue = deque()

        _regions = defaultdict(lambda: -1)
        sys_winning_region = []
        env_winning_region = []
        sys_str = defaultdict(lambda: -1)
        env_str = defaultdict(lambda: -1)

        accepting_states = self.game.get_accepting_states()

        for _s in accepting_states:
            queue.append(_s)
            _regions[_s] = "eve"
            sys_winning_region.append(_s)

            if self.game._graph.nodes[_s].get("player") == "eve":
                for _succ_s in self.game._graph.successors(_s):
                    sys_str[_s] = _succ_s
                    break

        while queue:
            _s = queue.popleft()

            for _pre_s in self.game._graph.predecessors(_s):
                if _regions[_pre_s] == -1:
                    if self.game._graph.nodes[_pre_s].get("player") == "eve":
                        queue.append(_pre_s)
                        _regions[_pre_s] = "eve"
                        sys_winning_region.append(_pre_s)
                        sys_str[_pre_s] = _s

                    elif self.game._graph.nodes[_pre_s].get("player") == "adam":
                        num_out_edges[_pre_s] -= 1

                        if num_out_edges[_pre_s] == 0:
                            queue.append(_pre_s)
                            _regions[_pre_s] = "eve"
                            sys_winning_region.append(_pre_s)

                    else:
                        warnings.warn(f"Please make sure that every node in the game is assigned a player."
                                      f"Currently node {_pre_s} does not have a player assigned to it")

        for _s in self.game._graph.nodes():
            if _regions[_s] != "eve":
                _regions[_s] = "adam"
                env_winning_region.append(_s)

                if self.game._graph.nodes[_s]["player"] == "adam":

                    for _successor in self.game._graph.successors(_s):
                        if _regions[_successor] != "eve":
                            env_str[_s] = _successor

        self._sys_str = sys_str
        self._env_str = env_str
        self._sys_winning_region = sys_winning_region
        self._env_winning_region = env_winning_region

    def _epsilon_greedy_choose_action(self,
                                      _human_state: tuple,
                                      epsilon: float, _absoring_states: List[tuple]) -> Tuple[tuple, bool]:
        """
        Choose an action according to epsilon greedy algorithm

        Using this policy we either select a random human action with epsilon probability and the human can select the
        optimal action (as given in the str dict if any) with 1-epsilon probability.

        This method returns a human action based on the above algorithm.
        :param _human_state: The current state in the game from which we need to pick an action
        :return: Tuple(next_state, flag) . flag = True if human decides to take an action.
        """

        # combine the system and the env str dict
        str_dict = {**self.sys_str, **self.env_str}

        if self.game.get_state_w_attribute(_human_state, "player") != "adam":
            warnings.warn("WARNING: Randomly choosing action for a non human state!")

        _next_states = [_next_n for _next_n in self.game._graph.successors(_human_state)]
        _did_human_move = False

        # rand() return a floating point number between [0, 1)
        if np.random.rand() < epsilon:
            _next_state: tuple = random.choice(_next_states)
        else:
            # if a node belongs to sys's winning region, it does have a strategy associated with it. For those nodes,
            # randomly, pick one
            try:
                _next_state: tuple = str_dict[_human_state]
            except:
                _next_state: tuple = random.choice(_next_states)

        if _next_state[0][1] != _human_state[0][1]:
            _did_human_move = True

        return _next_state, _did_human_move

    def get_pos_sequences(self, epsilon: float, debug: bool = False) -> List[Tuple[str, ndarray, int]]:
        """
        A helper method to return a list of actions (edge labels) associated with the strategy found
        :return: A sequence of position of the robot
        """

        _start_state = self.game.get_initial_states()[0][0]
        _total_human_intervention = _start_state[0][1]
        _accepting_states = self.game.get_accepting_states()
        _trap_states = self.game.get_trap_states()

        _absorbing_states = _accepting_states + _trap_states
        _visited_states = []
        _position_sequence = []

        _curr_sys_node = _start_state
        _next_env_node = self.sys_str[_curr_sys_node]

        # randomly choose an env edge from new_env_node
        # next_sys_node = random.choice([_n for _n in self.game._graph.successors(_next_env_node)])
        _next_sys_node = self._epsilon_greedy_choose_action(_next_env_node,
                                                           epsilon=epsilon,
                                                           _absoring_states=_absorbing_states)[0]


        x, y = _next_sys_node[0][0]
        _human_interventions: int = _total_human_intervention - _next_sys_node[0][1]

        next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)

        _visited_states.append(_curr_sys_node)
        _visited_states.append(_next_sys_node)

        _entered_absorbing_state = False

        while 1:
            _position_sequence.append(next_pos)

            _curr_sys_node = _next_sys_node
            _next_env_node = self.sys_str[_curr_sys_node]

            # randomly choose an env edge from new_env_node
            # next_sys_node = random.choice([_n for _n in self.game._graph.successors(next_env_node)])
            if _next_env_node not in _absorbing_states:
                _next_sys_node = self._epsilon_greedy_choose_action(_next_env_node,
                                                                    epsilon=epsilon,
                                                                    _absoring_states=_absorbing_states)[0]
            else:
                _next_sys_node = _next_env_node

            if _next_sys_node in _visited_states:
                break

            # if you enter a trap/ accepting state then do not add that transition in _pos_sequence
            elif _next_sys_node in _absorbing_states:
                _entered_absorbing_state = True
                break
            else:
                x, y = _next_sys_node[0][0]
                _human_interventions: int = _total_human_intervention - _next_sys_node[0][1]
                next_pos = ("rand", np.array([int(x), int(y)]), _human_interventions)
                _visited_states.append(_next_sys_node)

        if not _entered_absorbing_state:
            _position_sequence.append(next_pos)

        if debug:
            print([_n for _n in _position_sequence])

        return _position_sequence

    def print_winning_region(self):

        print("====================================")
        print(f"Winning Region for Sys player is : {[_n for _n in self.sys_winning_region]}")
        print("====================================")
        print(f"Winning Region for Env player is : {[_n for _n in self.env_winning_region]}")

    def print_winning_strategies(self):

        print("printing System player strategy")
        for _u, _v in self.sys_str.items():
            print(f"{_u} ------> {_v}")

        print("printing Env player strategy")
        for _u, _v in self.env_str.items():
            print(f"{_u} ------> {_v}")

    def is_winning(self) -> bool:
        """
        A helper method that return True if the initial state(s) belongs to the list of system player' winning region
        :return: boolean value indicating if system player can force a visit to the accepting states or not
        """
        _init_states = self.game.get_initial_states()

        for _i in _init_states:
            if _i[0] in self.sys_winning_region:
                return True
        return False

    def _sanity_check_total(self, debug: bool = False):
        """
        A helper method to add a self loop to a node that does not have any successors. This is done ensure
        that the two player graph is total.

        :param debug:
        :return:
        """
        for _n in self.game._graph.nodes():
            if len(list(self.game._graph.successors(_n))) == 0:
                if debug:
                    print("====================================")
                    print(f"Adding a self loop to state {_n}")
                    print("====================================")

                self.game._graph.add_edge(_n, _n)

    def _sanity_check_player(self):

        for _n in self.game._graph.nodes.data("player"):
            if _n[1] is None:
                self.game.add_state_attribute(_n[0], "player", "eve")


if __name__ == "__main__":

    debug = True

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="config/two_player_graph",
                                         save_flag=True,
                                         pre_built=False,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam)
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "adam")
    two_player_graph.add_state_attribute("s3", "player", "adam")
    two_player_graph.add_state_attribute("s4", "player", "eve")
    two_player_graph.add_state_attribute("s5", "player", "adam")
    two_player_graph.add_state_attribute("s6", "player", "eve")
    two_player_graph.add_state_attribute("s7", "player", "adam")

    two_player_graph.add_edge("s0", "s1")
    two_player_graph.add_edge("s0", "s3")
    two_player_graph.add_edge("s1", "s0")
    two_player_graph.add_edge("s1", "s2")
    two_player_graph.add_edge("s1", "s4")
    two_player_graph.add_edge("s2", "s2")
    two_player_graph.add_edge("s2", "s4")
    two_player_graph.add_edge("s3", "s4")
    two_player_graph.add_edge("s3", "s0")
    two_player_graph.add_edge("s3", "s5")
    two_player_graph.add_edge("s4", "s3")
    two_player_graph.add_edge("s4", "s1")
    two_player_graph.add_edge("s5", "s3")
    two_player_graph.add_edge("s5", "s6")
    two_player_graph.add_edge("s6", "s6")
    two_player_graph.add_edge("s6", "s7")
    two_player_graph.add_edge("s7", "s3")
    two_player_graph.add_edge("s7", "s0")

    two_player_graph.add_accepting_states_from(["s3", "s4"])

    reachability_game_handle = ReachabilityGame(game=two_player_graph, debug=debug)
    reachability_game_handle.reachability_solver()
    reachability_game_handle.print_winning_region()
    reachability_game_handle.print_winning_strategies()