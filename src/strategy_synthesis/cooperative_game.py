import warnings

from collections import defaultdict, deque

# import local packages
from ..graph import TwoPlayerGraph
from .adversarial_game import ReachabilityGame


class CooperativeGame(ReachabilityGame):
    """
        This class inherits ReachabilityGame class and modifies the reachability solver and the permissive reachability solver to compute set of states form which there exists a path to the accepting state(s).
    """

    def __init__(self, game: TwoPlayerGraph, debug: bool = False, extract_strategy: bool = False):
        super().__init__(game, debug)
        self.extract_strategy = extract_strategy

    def reachability_solver(self) -> None:

        queue = deque()

        _regions = defaultdict(lambda: -1)
        sys_winning_region = set({})
        env_winning_region = set({})
        if self.extract_strategy:
            sys_str = defaultdict(lambda: set({}))
            env_str = defaultdict(lambda: set({}))

        accepting_states = self.game.get_accepting_states()

        assert len(accepting_states) != 0, "For Value Iteration algorithm, you need atleast one accepting state. FIX THIS!!!"

        for _s in accepting_states:
            queue.append(_s)
            _regions[_s] = "eve"
            sys_winning_region.add(_s)
        
        while queue:
            # pop a state and look at it's predecessor
            _s = queue.popleft()
            
            for _pre_s in self.game._graph.predecessors(_s):
                if _regions[_pre_s] == -1:
                    queue.append(_pre_s)
                    _regions[_pre_s] = "eve"
                    sys_winning_region.add(_pre_s)
                    if self.extract_strategy:
                        if self.game._graph.nodes[_pre_s]["player"] == "eve":
                            sys_str[_pre_s].add(_s)
                        elif self.game._graph.nodes[_pre_s]["player"] == "adam":
                            env_str[_pre_s].add(_s)
                        else:
                            warnings.warn(f"Please make sure that every node in the game is assigned a player."
                                          f"Currently node {_pre_s} does not have a player assigned to it")
        

        # Note, as this is a min-min game, env's winning region belongs to a set of states from which there
        # does not exists a path to the accepting state 
        env_winning_region = self.game_states.difference(sys_winning_region)
        
        if self.extract_strategy:
            for _s in accepting_states:
                if self.game._graph.nodes[_s].get("player") == "eve":
                    for _succ_s in self.game._graph.successors(_s):
                        if _regions[_succ_s] == "eve":
                            sys_str[_s].add(_succ_s)
                            break
        
        self._sys_winning_region = sys_winning_region
        self._env_winning_region = env_winning_region
        if self.extract_strategy:
            _processed_sys_str = {}
            _processed_env_str = {}
            for states, winning_str in sys_str.items(): 
                _processed_sys_str[states] = list(winning_str) 
            
            for states, env_winning_str in env_str.items(): 
                _processed_env_str[states] = list(env_winning_str) 
            self._sys_str = _processed_sys_str
            self._env_str = _processed_env_str
    

    def plot_graph(self, with_strategy: bool = False) -> None:
        """
         Modifying the base method to add a label to the state to indicate which state belongs to the cooperative winning region.
        """

        # add state values of zero to all states that belong to the cooperative strongly winning region. 
        for _n in self._sys_winning_region:
            self.game.add_state_attribute(_n, "val", [0])

        if not with_strategy:
            self.game.plot_graph()

        else:
            assert self.extract_strategy is True, "Please Rerun the solve with extract_strategy flag set to True"
            assert len(self.sys_str.keys()) != 0, "A winning strategy does not exists. Did you run the solver?"

            self.game.set_edge_attribute('strategy', False)

            # adding attribute to winning strategy so that they are colored when plotting.
            for curr_node, next_node in self.sys_str.items():
                # if self.game._graph.nodes[curr_node].get("player") == "eve":
                if isinstance(next_node, list):
                    for n_node in next_node:
                        self.game._graph.edges[curr_node, n_node, 0]['strategy'] = True
                else:
                    self.game._graph.edges[curr_node, next_node, 0]['strategy'] = True
            self.game.plot_graph()