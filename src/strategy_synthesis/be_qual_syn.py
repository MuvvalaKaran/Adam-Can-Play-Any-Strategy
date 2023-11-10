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
        self._winning_region: Union[List, set] = None
        self._coop_winning_region: Union[List, set] = None
        self._losing_region: Union[List, set] = None
        self._pending_region: Union[List, set] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_coop_winning_str: Optional[dict] = None
        self._sys_best_effort_str: Optional[dict] = None

        self.game_states = set(self.game.get_states()._nodes.keys())
        self.debug: bool = debug
    

    @property
    def game(self):
        return self._game

    @property
    def winning_region(self):
        if not bool(self._winning_region):
            self.get_winning_region()
        return self._winning_region

    @property
    def losing_region(self):
        if not bool(self._losing_region):
            self.get_losing_region()
        
        return self._losing_region

    @property
    def pending_region(self):
        if not bool(self._pending_region):
            self.get_pending_region()
        return self._pending_region
    
    @property
    def coop_winning_region(self):
        return self._coop_winning_region

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

        assert isinstance(game, TwoPlayerGraph), "Please enter a graph which is of type TwoPlayerGraph"

        self._game = game
        self.game_states = set(game.get_states()._nodes.keys())
    

    def is_winning(self) -> bool:
        """
        A helper method that return True if the initial state(s) belongs to the list of system player' winning region
        :return: boolean value indicating if system player can force a visit to the accepting states or not
        """
        _init_states = self.game.get_initial_states()

        for _i in _init_states:
            if _i[0] in self.winning_region:
                return True
        return False
    

    def get_losing_region(self, print_states: bool = False):
        """
            A Method that compute the set of states from which there does not exist a path to the target state(s). 
        """
        assert bool(self._coop_winning_region) is True, "Please Run the solver before accessing the Losing region."
        self._losing_region = self.game_states.difference(self._coop_winning_region)

        if print_states:
            print("Losing Region: \n", self._losing_region)
    

    def get_pending_region(self, print_states: bool = False):
        """
            A Method that compute the set of states from which there does exists a path to the target state(s). 
        """
        assert bool(self._winning_region) is True, "Please Run the solver before accessing the Pending region."
        if not bool(self._losing_region):
            self._losing_region = self.game_states.difference(self._coop_winning_region)
        
        tmp_states = self._losing_region.union(self.winning_region)
        self._pending_region =  self.game_states.difference(tmp_states)

        if print_states:
            print("Pending Region: \n", self._pending_region)

    def get_winning_region(self, print_states: bool = False):
        """
            A Method that compute the set of states from which the sys player can enforce a visit to the target state(s). 
        """
        assert bool(self._winning_region) is True, "Please Run the solver before accessing the Winning region."
        
        if print_states:
            print("Winning Region: \n", self._winning_region)


    def compute_cooperative_winning_strategy(self):
        """
            A Method that computes the cooperatively winning strategy, cooperative winning region and Losing region.
        """
        coop_handle = CooperativeGame(game=self.game, debug=self.debug, extract_strategy=True)
        coop_handle.reachability_solver()
        self._sys_coop_winning_str = coop_handle.sys_str
        self._coop_winning_region = coop_handle.sys_winning_region
        
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
        self._winning_region = reachability_game_handle.sys_winning_region
        
        if self.debug and reachability_game_handle.is_winning:
            print("There exists a Winning strategy from the Initial State")


    def compute_best_effort_strategies(self, plot: bool = False):
        """
            This method calls compute_winning_strategies() and compute_cooperative_winning_strategy() methods and stitches them together. 
        """
        # get winning strategies
        self.compute_winning_strategies()

        # get cooperative winning strategies
        self.compute_cooperative_winning_strategy()

        # get sys states in winning region and remove those element from cooperative winning region
        _sys_coop_win_sys: dict = {s: strat for s, strat in self._sys_coop_winning_str.items() if self.game.get_state_w_attribute(s, 'player') is 'eve'}

        for winning_state in self.winning_region:
            if self.game.get_state_w_attribute(winning_state, 'player') is 'eve':
                try:
                    del _sys_coop_win_sys[winning_state]
                except:
                    print("[ERROR]: Encountered a state that exists in Winning region but does not exists in Cooperative Winning region. This is wrong")
                    sys.exit(-1)
        

        # merge the dictionaries
        self._sys_best_effort_str = {**self.sys_winning_str, **_sys_coop_win_sys}

        if plot:
            self.add_str_flag()
            self.game.plot_graph()


    def add_str_flag(self):
        """
        A helper function used to add the 'strategy' attribute to edges that belong to the winning strategy. 
        This function is called before plotting the winning strategy.
        """
        self.game.set_edge_attribute('strategy', False)

        for curr_node, next_node in self._sys_best_effort_str.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    self.game._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.game._graph.edges[curr_node, next_node, 0]['strategy'] = True