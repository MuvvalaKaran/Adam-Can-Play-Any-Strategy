import warnings

from typing import Optional, Iterable

from ..graph import TwoPlayerGraph

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
        self._sys_winning_region: Optional[Iterable] = None
        self._sys_losing_region: Optional[Iterable] = None
        self._sys_pending_region: Optional[Iterable] = None
        self._env_winning_region: Optional[Iterable] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_pending_str: Optional[dict] = None
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
    def sys_winning_str(self):
        return self._sys_winning_str

    @property
    def env_winning_str(self):
        return self._env_winning_str

    @property
    def sys_pending_str(self):
        return self._sys_pending_str

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
        raise NotImplementedError
    

    def compute_winning_strategies(self):
        """
            A Method that computes the Winning strategies and corresponding winning region
        """
        raise NotImplementedError


    def compute_best_effort_strategies(self):
        """
            This method calls compute_winning_strategies() and compute_cooperative_winning_strategy() methods and stitches them together. 
        """
        raise NotImplementedError


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