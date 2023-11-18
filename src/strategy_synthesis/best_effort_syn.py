import sys
import math
import copy
import warnings

from collections import defaultdict
from typing import Optional, Union, List, Iterable, Dict

from ..graph import TwoPlayerGraph
from .adversarial_game import ReachabilityGame
from .cooperative_game import CooperativeGame
from .value_iteration import ValueIteration, PermissiveValueIteration, PermissiveSafetyValueIteration


class QualitativeBestEffortReachSyn():
    """
    Class that computes Qualitative Best Effort strategies with reachability objectives. 
    
    The algorithm is as follows:

    1. Given a target set, identify Winning region and synthesize winning strategies.
    2. Given a target set, identify Cooperatively Winning (Pending) region and  synthesize cooperative winning strategies
    3. Merge the strategies. 
        3.1 States that belong to winning region play the winning strategy
        3.2 States that belong to pending region play cooperative winning strategy
        3.3 States that belong to losing region play any strategy
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        self._game = game
        self._winning_region: Union[List, set] = set({})
        self._coop_winning_region: Union[List, set] = None
        self._losing_region: Union[List, set] = None
        self._pending_region: Union[List, set] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_coop_winning_str: Optional[dict] = None
        self._sys_best_effort_str: Optional[dict] = None
        self._best_effort_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._coop_winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)

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

    @property
    def best_effort_state_values(self):
        return self._best_effort_state_values
    
    @property
    def winning_state_values(self):
        return self._winning_state_values

    @property
    def coop_winning_state_values(self):
        return self._coop_winning_state_values

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
        
        return self._losing_region
    

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
        
        return self._pending_region

    def get_winning_region(self, print_states: bool = False):
        """
        A Method that compute the set of states from which the sys player can enforce a visit to the target state(s). 
        """
        assert bool(self._winning_region) is True, "Please Run the solver before accessing the Winning region."
        
        if print_states:
            print("Winning Region: \n", self._winning_region)
        
        return self._winning_region

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
    

    def compute_winning_strategies(self, permissive: bool = False):
        """
        A Method that computes the Winning strategies and corresponding winning region.
         Set the permissive flag to True to compute the set of all winning strategies. 
        """
        reachability_game_handle = ReachabilityGame(game=self.game, debug=self.debug)
        reachability_game_handle.reachability_solver()
        self._sys_winning_str = reachability_game_handle.sys_str
        self._env_winning_str = reachability_game_handle.env_str

        # sometime an accepting may not have a winning strategy. Thus, we only store states that have an winning strategy
        _sys_states_winning_str = reachability_game_handle.sys_str.keys()

        for ws in reachability_game_handle.sys_winning_region:
            if self.game.get_state_w_attribute(ws, 'player') == 'eve' and ws in _sys_states_winning_str:
                self._winning_region.add(ws)
            elif self.game.get_state_w_attribute(ws, 'player') == 'adam':
                self._winning_region.add(ws)
        
        if self.debug and reachability_game_handle.is_winning:
            print("There exists a Winning strategy from the Initial State")


    def compute_best_effort_strategies(self, plot: bool = False, permissive: bool = False):
        """
        This method calls compute_winning_strategies() and compute_cooperative_winning_strategy() methods and stitches them together. 
        """
        # get winning strategies
        self.compute_winning_strategies(permissive=permissive)

        # get cooperative winning strategies
        self.compute_cooperative_winning_strategy()

        # get sys states in winning region and remove those element from cooperative winning region
        _sys_coop_win_sys: dict = {s: strat for s, strat in self._sys_coop_winning_str.items() if self.game.get_state_w_attribute(s, 'player') == 'eve'}

        for winning_state in self.winning_region:
            if self.game.get_state_w_attribute(winning_state, 'player') == 'eve':
                try:
                    del _sys_coop_win_sys[winning_state]
                except KeyError:
                    warnings.warn("[ERROR]: Encountered a state that exists in Winning region but does not exists in Cooperative Winning region. This is wrong! \
                                  state {ps} does not exists in BE Safety and BE Reachability strategy dictionary!")
                    sys.exit(-1)
        
        # for states that belong to the losing region, we can play any strategy
        _sys_losing_str = {state: list(self.game._graph.successors(state)) for state in self.get_losing_region() if self.game.get_state_w_attribute(state, 'player') == 'eve'} 

        # merge the dictionaries
        self._sys_best_effort_str = {**self.sys_winning_str, **_sys_coop_win_sys, **_sys_losing_str}

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
            if isinstance(next_node, set) or isinstance(next_node, list):
                for n_node in next_node:
                    self.game._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.game._graph.edges[curr_node, next_node, 0]['strategy'] = True



class QualitativeBestEffortSafetySyn():
    """
    This class implements best-effort safety synthesis algorithm. Given, a two-player game, a set of target states,
      compute Best-effort safety strategies that ensures the robot is doing its best to stay within the safe region (target states).
    
    Intuitively, Best-effort safety is a weaker form of safety game, where set of states in the safety game are states from which the robot can
      enforce staying in the safe region. A safe region is a set of states that do not belong to losing region (or belong to Winning + Pending Region)
    """


    def __init__(self, game: TwoPlayerGraph, target_states: Iterable, debug: bool = False) -> None:
        self._game = game
        self._winning_region: Union[List, set] = set({})
        self._coop_winning_region: Union[List, set] = None
        self._losing_region: Union[List, set] = None
        self._pending_region: Union[List, set] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_coop_winning_str: Optional[dict] = None
        self._sys_best_effort_str: Optional[dict] = None
        self._best_effort_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._coop_winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)

        self.game_states = set(self.game.get_states()._nodes.keys())
        self.debug: bool = debug
        self.target_states = target_states

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
    

    @property
    def best_effort_state_values(self):
        return self._best_effort_state_values
    
    @property
    def winning_state_values(self):
        return self._winning_state_values

    @property
    def coop_winning_state_values(self):
        return self._coop_winning_state_values
    

    def get_losing_region(self, print_states: bool = False):
        """
        A Method that compute the set of states from which there does not exist a strategy in the safe game. 
        """
        _losing_region = self.game_states.difference(self._sys_winning_region)
        self._losing_region = [state for state in _losing_region if self.game.get_state_w_attribute(state, 'player') == 'eve']
        if print_states:
            print("Losing Region: \n", self._losing_region)
        
        return self._losing_region
    

    def compute_cooperative_winning_strategy(self):
        """
        A Method that computes the cooperatively winning strategy, cooperative winning region and Losing region.

        Approach : Convert the original game into a Quantitative game with unit edge weight everywhere. 
        Then we play Quantitative Min-Min game. Note: The edge weight added because Value Iteration takes in a wieght graph.
        The edge weights do not play any role in Value iterartion. Check Algorithm for more info. 
        """
        tmp_copy_game = copy.deepcopy(self.game)
        # create a local copy of the game and modify the accpeting states
        tmp_copy_game.add_accepting_states_from([state for state in self.target_states if self.game.get_state_w_attribute(state, "player") == 'eve'])
        for _s in tmp_copy_game._graph.nodes():
            for _e in tmp_copy_game._graph.out_edges(_s):
                tmp_copy_game._graph[_e[0]][_e[1]][0]["weight"] = 1 if tmp_copy_game._graph.nodes(data='player')[_s] == 'eve' else 0

        coop_handle = PermissiveSafetyValueIteration(game=tmp_copy_game, competitive=True)
        coop_handle.solve(debug=False, plot=False, extract_strategy=True)
        self._sys_winning_str = coop_handle.sys_str_dict
        self._sys_winning_region =  set(coop_handle.sys_str_dict.keys())
        self._sys_coop_winning_str = coop_handle.sys_str_dict
        self._coop_winning_region = set(coop_handle.sys_str_dict.keys())
        # self._pending_region = self._coop_winning_region
        
        if self.debug and coop_handle.is_winning():
            print("There exists a path from the Initial State")


    def compute_best_effort_safety_strategies(self, plot: bool = False):
        """
        This method converts the safety game into a Reachability game by assigning all the target states as the accepting states
           and the objective of the sys player is visit the accepting states. 
        """
        # create a reacability game and then compute BE reachability strategies.
        self.compute_cooperative_winning_strategy()

        # for states that belong to the losing region, we can play any strategy
        _sys_losing_str = {state: list(self.game._graph.successors(state)) for state in self.get_losing_region() if self.game.get_state_w_attribute(state, 'player') == 'eve'} 

        # update dictionary 
        self._sys_best_effort_str = {**self.sys_winning_str, **_sys_losing_str}

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
            if isinstance(next_node, set) or isinstance(next_node, list):
                for n_node in next_node:
                    self.game._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.game._graph.edges[curr_node, next_node, 0]['strategy'] = True



class QuantitativeBestEffortReachSyn(QualitativeBestEffortReachSyn):
    """
    Class that computes Quantitative Best Effort strategies with reachability objectives. 
    
    The algorithm is as follows:

    1. Given a target set, identify Winning region and synthesize winning strategies.
    2. Given a target set, identify Cooperatively Winning (Pending) region and synthesize cooperative winning strategies
    3. Merge the strategies. 
        3.1 States that belong to winning region play the winning strategy
        3.2 States that belong to pending region play cooperative winning strategy
        3.3 States that belong to losing region play any strategy
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        super().__init__(game, debug)
    

    def _add_state_costs_to_graph(self):
        """
        A helper method that adds the Best Effort costs associated with each state.
        """
        for state in self.game._graph.nodes():
            self.game.add_state_attribute(state, "val", [self.best_effort_state_values[state]])

    

    def compute_cooperative_winning_strategy(self, permissive: bool = False):
        """
        Override the base method to run the Value Iteration code
        """
        coop_handle = PermissiveValueIteration(game=self.game, competitive=False)
        coop_handle.solve(debug=False, plot=True, extract_strategy=True)
        self._sys_coop_winning_str = coop_handle.sys_str_dict
        # self._coop_winning_region = (coop_handle.sys_winning_region).union(set(coop_handle.env_str_dict.keys()))
        self._coop_winning_region = set(coop_handle.sys_str_dict.keys()).union(set(coop_handle.env_str_dict.keys()))
        self._coop_winning_state_values = coop_handle.state_value_dict
        
        if self.debug and coop_handle.is_winning():
            print("There exists a path from the Initial State")


    def compute_winning_strategies(self, permissive: bool = False):
        """
        Override the base method to run the Value Iteration code
        """
        if permissive:
            reachability_game_handle = PermissiveValueIteration(game=self.game, competitive=True)
        else:    
            reachability_game_handle = ValueIteration(game=self.game, competitive=True)
        
        reachability_game_handle.solve(debug=False, plot=True, extract_strategy=True)
        self._sys_winning_str = reachability_game_handle.sys_str_dict
        self._env_winning_str = reachability_game_handle.env_str_dict
        self._winning_state_values = reachability_game_handle.state_value_dict

        # sometime an accepting may not have a winning strategy. Thus, we only store states that have an winning strategy
        _sys_states_winning_str = reachability_game_handle.sys_str_dict.keys()

        # update winning region and optimal state values
        for ws in reachability_game_handle.sys_winning_region:
            if self.game.get_state_w_attribute(ws, 'player') == 'eve' and ws in _sys_states_winning_str:
                self._winning_region.add(ws)
            elif self.game.get_state_w_attribute(ws, 'player') == 'adam':
                self._winning_region.add(ws)
        
        if self.debug and reachability_game_handle.is_winning():
            print("There exists a Winning strategy from the Initial State")


class QuantitativeBestEffortSafetySyn(QualitativeBestEffortSafetySyn):
    """
     This class implements best-effort safety synthesis algorithm with quantitative objectives.
       Given, a two-player game, a set of target states, compute Best-effort sfaty strategies that ensures the robot is doing its best to stay within the safe region (target states).

     The Algorithm is same as in QualitativeBestEffortSafetySyn class.
    """


    def __init__(self, game: TwoPlayerGraph, target_states: Iterable, debug: bool = False) -> None:
        super().__init__(game, target_states, debug)


    # def compute_best_effort_safety_strategies(self, plot: bool = False):
    #     """
    #      This methods converts the safety games into a Reachability game by assigning all the target states as the accepting states
    #        and the objective of the sys player is visit the accepting states. 
    #     """

    #     # create a reacability game and then compute BE reachability strategies.

    #     # create a local copy of the game and modify the accpeting states
    #     game_copy = copy.deepcopy(self.game)
    #     game_copy.add_accepting_states_from(self.target_states)

    #     best_effort_reach_handle = QuantitativeBestEffortReachSyn(game=game_copy, debug=self.debug)
    #     best_effort_reach_handle.compute_best_effort_strategies(plot=plot, permissive=True)

    #     # update dictionaries 
    #     self._sys_best_effort_str = best_effort_reach_handle.sys_best_effort_str
    #     self._sys_winning_str = best_effort_reach_handle.sys_winning_str
    #     self._env_winning_str = best_effort_reach_handle.env_winning_str
    #     self._sys_coop_winning_str = best_effort_reach_handle.sys_coop_winning_str

    #     # update regions
    #     self._winning_region = best_effort_reach_handle.winning_region
    #     self._coop_winning_region = best_effort_reach_handle.coop_winning_region

    #     self._pending_region = best_effort_reach_handle.get_pending_region()
        # self._losing_region = best_effort_reach_handle.get_losing_region()