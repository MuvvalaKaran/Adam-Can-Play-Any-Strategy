import sys
import warnings

from typing import Optional, Union, List


from ..graph import TwoPlayerGraph

from .adversarial_game import ReachabilityGame
from .cooperative_game import CooperativeGame

class QualBestEffortReachabilitySynthesis():
    """
    Class that computes Qualitative Best Effort strategies with reachability objectives. 
    
    The algorithm is as follows:

    2. Given a target set, identify Winning region and synthesize winning strategies.
    3. Given, a target set, identify Cooperatively Winning (Pending) region synthesize cooperative winning strategies
    3. Merge the strategies. 
        3.1 States that belong to winning region play the winning strategy
        3.2 States that belong to pending region play cooperative winning strategy
        3.3 States that belong to losing region play any strategy
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        self._game = game
        self.debug: bool = debug
        self._sys_winning_region: Union[List, set] = None
        self._sys_coop_winning_region: Union[List, set] = None
        self._sys_losing_region: Union[List, set] = None
        self._sys_pending_region: Union[List, set] = None
        self._env_winning_region: Union[List, set] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_coop_winning_str: Optional[dict] = None
        self._sys_best_effort_str: Optional[dict] = None
    

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
    def sys_losing_region(self):
        return self._sys_losing_region

    @property
    def sys_pending_region(self):
        return self._sys_pending_region
    
    @property
    def sys_coop_winning_region(self):
        return self._sys_coop_winning_region

    @property
    def sys_winning_str(self):
        return self._sys_winning_str

    @property
    def env_winning_str(self):
        return self._env_winning_str

    @property
    def sys_coop_winning_str(self):
        return self._sys_coop_winning_str

    @property
    def sys_best_effort_str(self):
        return self._sys_best_effort_str

    @game.setter
    def game(self, game: TwoPlayerGraph):

        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = game


    def compute_cooperative_winning_strategy(self):
        """
            A Method that computes the cooperatively winning strategy, cooperative winning region and Losing region.
        """
        coop_handle = CooperativeGame(game=self.game, debug=self.debug, extract_strategy=True)
        coop_handle.reachability_solver()
        self._sys_coop_winning_str = coop_handle.sys_str
        self._sys_coop_winning_region = coop_handle.sys_winning_region
        
        if self.debug and coop_handle.is_winning:
            print("There exists a path from the Initial State")
    

    def compute_winning_strategies(self):
        """
            A Method that computes the Winning strategies and corresponding winning region. 
        """
        reachability_game_handle = ReachabilityGame(game=self.game, debug=self.debug)
        reachability_game_handle.maximally_permissive_reachability_solver()
        self._sys_winning_str = reachability_game_handle.sys_str
        self._env_winning_str = reachability_game_handle.env_str
        self._sys_winning_region = reachability_game_handle.sys_winning_region
        
        if self.debug and reachability_game_handle.is_winning:
            print("There exists a Winning strategy from the Initial State")


    def compute_best_effort_strategies(self):
        """
            This method calls compute_winning_strategies() and compute_cooperative_winning_strategy() methods and stitches them together. 
        """
        # get winning strategies
        self.compute_winning_strategies()

        # get cooperative winning strategies
        self.compute_cooperative_winning_strategy()

        # get states in winning region and remove those element from cooperative winning region
        _sys_coop_win_sys: dict = self.sys_coop_winning_str

        for winning_state in self.sys_winning_region:
            if winning_state in _sys_coop_win_sys: 
                del _sys_coop_win_sys[winning_state]
            else:
                print("[ERROR]: Encountered a state that exists in Winnig region but does not exists in Cooperative Winning region. This is wrong")
                sys.exit(-1)
        

        # merge the dictionaries
        self._sys_best_effort_str = {**self.sys_winning_str, **_sys_coop_win_sys}
        tmp_sys_best_effort_str = {}

        
        # loop over every state and merge strategies
        for state, state_attr in self.game.get_states():
            # if state_attr['player'] == 'eve':
            if state in self.sys_winning_region:
                # print(f"{state} in Winning Region")
                tmp_sys_best_effort_str[state] = self.sys_winning_str[state]
            elif state in self.sys_coop_winning_region:
                # print(f"{state} in Cooperative Winning Region")
                tmp_sys_best_effort_str[state] = self.sys_coop_winning_str[state]
                # else:
                #     # print(f"{state} in Losing Region")
        
        # sanity checking
        print(f"Sanity Check Passed? {tmp_sys_best_effort_str == self.sys_best_effort_str}")



    def add_str_flag(self):
        """

        :param str_dict:
        :return:
        """
        self.game.set_edge_attribute('strategy', False)

        for curr_node, next_node in self._sys_best_effort_str.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    self.org_graph._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.org_graph._graph.edges[curr_node, next_node, 0]['strategy'] = True


    def plot_graph(self):
        """
         A helper function that changes the original name of the graph, add state cost as attribute, adds strategy
           flags to strategies and
        """
        self.game.plot_graph()