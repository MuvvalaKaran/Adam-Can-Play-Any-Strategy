from ..graph import TwoPlayerGraph
from .value_iteration import ValueIteration, PermissiveValueIteration
from .be_qual_syn import QualBestEffortReachabilitySynthesis


class QuantBestEffortReachabilitySynthesis(QualBestEffortReachabilitySynthesis):
    """
    Class that computes Quantitative Best Effort strategies with reachability objectives. 
    
    The algorithm is as follows:

    2. Given a target set, identify Winning region and synthesize winning strategies.
    3. Given, a target set, identify Cooperatively Winning (Pending) region synthesize cooperative winning strategies
    3. Merge the strategies. 
        3.1 States that belong to winning region play the winning strategy
        3.2 States that belong to pending region play cooperative winning strategy
        3.3 States that belong to losing region play any strategy
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> None:
        super().__init__(game, debug)
    

    def compute_cooperative_winning_strategy(self):
        """
        Override the base method to run the Value Iteration code
        """
        coop_handle = PermissiveValueIteration(game=self.game, competitive=False)
        coop_handle.solve()

        self._sys_coop_winning_str = coop_handle.sys_str_dict
        self._coop_winning_region = coop_handle.sys_winning_region
        
        if self.debug and coop_handle.is_winning():
            print("There exists a path from the Initial State")


    def compute_winning_strategies(self):
        """
        Override the base method to run the Value Iteration code
        """
        reachability_game_handle = ValueIteration(game=self.game, competitive=True)
        reachability_game_handle.solve()
        self._sys_winning_str = reachability_game_handle.sys_str_dict
        self._env_winning_str = reachability_game_handle.env_str_dict
        self._winning_region = reachability_game_handle.sys_winning_region
        
        if self.debug and reachability_game_handle.is_winning():
            print("There exists a Winning strategy from the Initial State")
