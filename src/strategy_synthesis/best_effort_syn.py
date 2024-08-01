import sys
import time
import math
import copy
import operator
import warnings
import numpy as np
import networkx as nx

from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from typing import Optional, Union, List, Iterable, Dict, Set, Tuple, Generator

from ..graph import TwoPlayerGraph
from ..graph import graph_factory
from .adversarial_game import ReachabilityGame
from .cooperative_game import CooperativeGame
from .value_iteration import ValueIteration, PermissiveValueIteration, HopefulPermissiveValueIteration


class AbstractBestEffortReachSyn(metaclass=ABCMeta):
    """
     A abstract class for BestEffort and Admissibility Synthesis algorithms.

     This is the correct way to create meta classes in Python3. 
     See Reference:https://stackoverflow.com/questions/7196376/python-abstractmethod-decorator
    """
    def __init__(self,  game: TwoPlayerGraph, debug: bool = False):
        super().__init__()
        self._game = game
        self._num_of_nodes: int = len(list(self.game._graph.nodes))
        self._winning_region: Union[List, set] = set({})
        self._coop_winning_region: Union[List, set] = None
        self._losing_region: Union[List, set] = None
        self._pending_region: Union[List, set] = None
        self._sys_winning_str: Optional[dict] = None
        self._env_winning_str: Optional[dict] = None
        self._sys_coop_winning_str: Optional[dict] = None
        self._env_coop_winning_str: Optional[dict] = None
        self._sys_best_effort_str: Optional[dict] = None
        self._env_best_effort_str: Optional[dict] = None
        self._best_effort_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._coop_winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)

        self.debug: bool = debug
        self.game_init_states: List = self.game.get_initial_states()
        self.game_states: Set[str] = set(self.game.get_states()._nodes.keys())
        self.target_states: Set[str] = set(s for s in self.game.get_accepting_states())
    

    @property
    def game(self):
        return self._game
    
    @property
    def num_of_nodes(self):
        return self._num_of_nodes

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
    def env_coop_winning_str(self):
        return self._env_coop_winning_str

    @property
    def sys_best_effort_str(self):
        return self._sys_best_effort_str
    
    @property
    def env_best_effort_str(self):
        return self._env_best_effort_str

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
        for _i in self.game_init_states:
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
    
    @abstractmethod
    def compute_cooperative_winning_strategy(self):
        raise NotImplementedError
    
    @abstractmethod
    def compute_winning_strategies(self):
        raise NotImplementedError
    
    @abstractmethod
    def compute_best_effort_strategies(self):
        raise NotImplementedError



class QualitativeBestEffortReachSyn(AbstractBestEffortReachSyn):
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

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> 'QualitativeBestEffortReachSyn':
        super().__init__(game=game, debug=debug)
    

    def compute_cooperative_winning_strategy(self):
        """
        A Method that computes the cooperatively winning strategy, cooperative winning region and Losing region.
        """
        coop_handle = CooperativeGame(game=self.game, debug=self.debug, extract_strategy=True)
        coop_handle.reachability_solver()
        self._sys_coop_winning_str = coop_handle.sys_str
        self._env_coop_winning_str = coop_handle.env_str
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



class QuantitativeBestEffortReachSyn(AbstractBestEffortReachSyn):
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

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> 'QuantitativeBestEffortReachSyn':
        super().__init__(game, debug)
        self.sccs: List[List] = None

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
        coop_handle.solve(debug=False, plot=False, extract_strategy=True)
        self._sys_coop_winning_str = coop_handle.sys_str_dict
        self._env_coop_winning_str = coop_handle.env_str_dict
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
        
        reachability_game_handle.solve(debug=False, plot=False, extract_strategy=True)
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

    def states_in_same_scc(self, state1, state2) -> bool:
        """
         A helper method to check if two state belong to same SCCs or not.
        """
        for scc in self.sccs:
            if state1 in scc and state2 in scc:
                return True
        return False
    
    def check_if_play_loops(self, start_state: str, end_state: str, play: Set[str] = set({}), sanity_check: bool = False, play_len: int = 0) -> bool:
        """
         A helper function that checks if the play starting from start_state induced by the optimal winning strategy or cooperative winning strategy starting loops. 
        """
        curr_state = start_state
        play.add(start_state)
        
        while end_state not in play and not any(state in play for state in self.target_states):
            if curr_state in self.pending_region:
               next_state: List[str] = self.sys_coop_winning_str[curr_state] if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else self.env_coop_winning_str[curr_state]
            else:
                next_state: List[str] = self.sys_winning_str[curr_state]
            
            # preprocess next_state to be if type list
            if not isinstance(next_state, list):
                next_state = [next_state] 

            # all of this is correct if len(next_state) == 1
            if len(next_state) == 1:
                if next_state[0] == end_state:
                    return True
                elif next_state[0] in self.target_states:
                    return False
                
                # sanity checking
                if sanity_check and play_len > self.num_of_nodes:
                    warnings.warn(f"[Error] Problem rolling out the optimal strategy from state: {start_state}. The length of play exceeds the max length of {self.num_of_nodes}")
                    sys.exit(-1)
            
                play.add(next_state[0])
                play_len += 1 
                curr_state = next_state[0]
            else:
                # recursion, for all optimal strategies
                all_plays_loops = np.zeros(len(next_state))
                for iter_count, ns in enumerate(next_state):
                    play_loops = self.check_if_play_loops(start_state=ns,
                                                          end_state=end_state,
                                                          play=copy.deepcopy(play),
                                                          play_len=play_len,
                                                          sanity_check=sanity_check)

                    if play_loops:
                        all_plays_loops[iter_count] = 1
                    
                    # the play does not loop; break the recursion
                    else:
                        return False
                        
                
                if all_plays_loops.all():
                    return True
        
        if end_state in play:
            return True

        return False
    
    def admissibility_check(self, curr_state , succ_state) -> None:
        """
         A method that implements the admissibility check for evert valid action from a state in winning and pending region.
        """
        # if the successor state belongs to the winning/cooperative region, then add the action to the admissible strategy
        if self.sys_winning_str.get(curr_state, None) == succ_state or self.sys_coop_winning_str.get(curr_state, None) == succ_state:
            self._sys_best_effort_str[curr_state].add(succ_state)
        
        # else first check if v, v' belong to the same SCC
        else:
            is_same_scc: bool = self.states_in_same_scc(curr_state, succ_state)
            if not is_same_scc:
                if self.game.get_edge_weight(curr_state, succ_state) + self.coop_winning_state_values[succ_state] < self.winning_state_values[curr_state]:
                    self._sys_best_effort_str[curr_state].add(succ_state)
            
            elif is_same_scc:
                # check if it is a loop, if yes, then ignore else, check if w(v, v') + cVal^{v'} < aVal^{v
                if not self.check_if_play_loops(start_state=succ_state, end_state=curr_state, play=set({}), sanity_check=True):
                    if self.game.get_edge_weight(curr_state, succ_state) + self.coop_winning_state_values[succ_state] < self.winning_state_values[curr_state]:
                        self._sys_best_effort_str[curr_state].add(succ_state)

    def compute_best_effort_strategies(self, plot: bool = False, permissive: bool = False):
        """
        This algorithm implements synthesis of Admissible Strategy as per the def in the following paper:

         Brenguier, Romain, et al. "Admissibility in quantitative graph games." arXiv preprint arXiv:1611.08677 (2016).

        The algorithm is as follows:
         1. Compute SCC using Tarjan's or Kosaraju Algorithm 
         2. Compute aVal and cVal using the standard VI algorithms and the corresponding optimal strategies
         3. For every state (v) that belongs to the Sys player and every valid action from v, add the action to the set of admissible strategy if:
            3.1 the state belongs to the Losing region
            3.2 The action belongs to Winning or Cooperatively winning strategy
            3.3 if v, v' belong to the same SCC then check if they loop. 
                3.3.1 If yes, then do NOT add the strategy
                3.3.2 If no, then check if w(v, v') + cVal^{v'} < aVal^{v}
            3.4 if v, v' do NOT belong to the same SCC then check if w(v, v') + cVal^{v'} < aVal^{v}
         4. Return the set of admissible strategies
        """
        # compute SCC
        self.sccs = list(s for s in nx.strongly_connected_components(self.game._graph))
        for scc in self.sccs:
            # print(type(scc))
            print(scc)
        
        # override best-effrot strategy to be a mappging from state to set of admissible strategies
        self._sys_best_effort_str = defaultdict(lambda: set({}))
        
        # get aVal and winning strategies, set permissive to True
        self.compute_winning_strategies(permissive=permissive)

        # get cVal and cooperative winning strategies
        self.compute_cooperative_winning_strategy()

        for state, s_attr in self.game.get_states_w_attributes():
            if s_attr['player'] == 'eve':
                if state in self.get_losing_region():
                    # add all the actions to the admissible strategy
                    self._sys_best_effort_str[state] = self._sys_best_effort_str[state].union(set(list(self.game._graph.successors(state))))
                else:
                    for succ_state in self.game._graph.successors(state):
                        self.admissibility_check(curr_state=state, succ_state=succ_state)
        
        if plot:
            self.add_str_flag()
            self.game.plot_graph()



class QuantitativeHopefullAdmissibleReachSyn(AbstractBestEffortReachSyn):

    def __init__(self, game: TwoPlayerGraph, debug: bool = False) -> 'QuantitativeHopefullAdmissibleReachSyn':
        super().__init__(game=game, debug=debug)
        self._adm_tree: TwoPlayerGraph = graph_factory.get("TwoPlayerGraph",
                                                           graph_name="adm_tree",
                                                           config_yaml="config/adm_tree",
                                                           save_flag=True,
                                                           from_file=False, 
                                                           plot=False)


    @property
    def adm_tree(self):
        if len(self._adm_tree._graph) == 0:
            warnings.warn("[Error] Got empty Tree. Please run best-effort synthesis code to construct tree")
            sys.exit(-1)
        return self._adm_tree


    def compute_cooperative_winning_strategy(self, permissive: bool = False, plot: bool = False):
        """
        Override the base method to run the Value Iteration code
        """
        coop_handle = PermissiveValueIteration(game=self.game, competitive=False)
        coop_handle.solve(debug=False, plot=plot, extract_strategy=True)
        self._sys_coop_winning_str = coop_handle.sys_str_dict
        self._env_coop_winning_str = coop_handle.env_str_dict
        # self._coop_winning_region = (coop_handle.sys_winning_region).union(set(coop_handle.env_str_dict.keys()))
        self._coop_winning_region = set(coop_handle.sys_str_dict.keys()).union(set(coop_handle.env_str_dict.keys()))
        self._coop_winning_state_values = coop_handle.state_value_dict
        
        if self.debug and coop_handle.is_winning():
            print("There exists a path from the Initial State")
    
    
    def compute_winning_strategies(self, permissive: bool = False, plot: bool = True):
        """
         Override the base method to run the Value Iteration code
        """
        if permissive:
            reachability_game_handle = PermissiveValueIteration(game=self.game, competitive=True)
        else:    
            reachability_game_handle = ValueIteration(game=self.game, competitive=True)
        
        reachability_game_handle.solve(debug=False, plot=plot, extract_strategy=True)
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
    

    def preprocess_org_graph(self, verbose: bool = False) -> TwoPlayerGraph:
        """
         This method preprocesses the graph to 
         1) Identify Losing Region
         2) Remove the Losing Region
         3) Remoce edges from states in the remaining grapg to Losing Region.
        Finally, for checking loops along a play, it convenient, we the state have unique integer number. We can do this by create bidict.
        """
        raise NotImplementedError

    def add_str_flag(self, adm_tree):
        """
        A helper function used to add the 'strategy' attribute to edges that belong to the winning strategy. 
        This function is called before plotting the winning strategy.
        """
        adm_tree.set_edge_attribute('strategy', False)

        for curr_node, next_node in self._sys_best_effort_str.items():
            if isinstance(next_node, set) or isinstance(next_node, list):
                for n_node in next_node:
                    adm_tree._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                adm_tree._graph.edges[curr_node, next_node, 0]['strategy'] = True
    

    def add_edges(self, ebunch_to_add, **attr) -> None:
        """
         A function to add all the edges in the ebunch_to_add. 
        """
        for e in ebunch_to_add:
            # u and v as 2-Tuple with node and node attributes of source (u) and destination node (v)
            u, v, dd = e
            u_node = u[0]
            u_attr = u[1]
            v_node = v[0]
            v_attr = v[1]
           
            key = None
            ddd = {}
            ddd.update(attr)
            ddd.update(dd)

            # add node attributes too
            self._adm_tree._graph.add_node(u_node, **u_attr)
            self._adm_tree._graph.add_node(v_node, **v_attr)
            
            # add edge attributes too 
            if not self._adm_tree._graph.has_edge(u_node, v_node):
                key = self._adm_tree._graph.add_edge(u_node, v_node, key)
                self._adm_tree._graph[u_node][v_node][key].update(ddd)
    

    def construct_tree(self, terminal_state_name: str = "vT",) -> Generator[Tuple, None, None]:
        """
         This method constructs tree in a non-recurisve depth first fashion. The worst case complexity is O(|E|).
        """
        max_depth: int = len(self.game._graph)
        source = self.game_init_states[0][0]
        nodes = [source]

        visited = set()
        for start in nodes:
            if start in visited:
                continue

            visited.add(start)
            stack = [(start, 0, iter(self.game._graph[start]))]
            stack_state_only = [start]
            depth_now: int = 1

            while stack:
                parent, pweight, children = stack[-1]
                for child in children:
                    if child not in stack_state_only:
                        cweight = pweight + self.game._graph[parent][child][0]['weight']
                        if child in  self.target_states:
                            yield ((parent, pweight), self.game._graph.nodes[parent]), ((child, cweight), self.game._graph.nodes[child]), {'weight': cweight}
                        else:
                            yield ((parent, pweight), self.game._graph.nodes[parent]), ((child, cweight), self.game._graph.nodes[child]), {'weight': 0}
                        visited.add(child)
                        if depth_now < max_depth:
                            stack.append((child, cweight, iter(self.game._graph[child])))
                            stack_state_only.append(child)
                            depth_now += 1
                            break
                    # a state that was already in play is visited. Add edge to terminal state.
                    else:
                        if parent not in self.target_states:
                            yield ((parent, pweight), self.game._graph.nodes[parent]), (terminal_state_name, {}), {'weight': 0}
                        else:
                            yield ((parent, pweight), self.game._graph.nodes[parent]), ((parent, pweight), self.game._graph.nodes[parent]), {'weight': 0}
                else:
                    stack.pop()
                    stack_state_only.pop()
                    depth_now -= 1
        
        # add self loop to terminal state
        yield (terminal_state_name, {}), (terminal_state_name, {}), {'weight': 0}

    
    def compute_best_effort_strategies(self, plot: bool = False):
        """
         In this algorithm we call the modified Value Iteration algorithm to computer permissive Admissible strategies.

         We need to first preprocee the graph, identify losing region, remove states that belong to losing region,
           and edges that transit to losing region. 

         The algorithm is as follows:
            First, we create a tree of bounded depth (|V| - 1)
            Then, we run the modified VI algorithm where the max node is looks at the second worst cost.
            Finally, we return the strategies. 
        """
        # first preprocess

        # now construct tree
        start = time.time()
        terminal_state: str = "vT"
        self._adm_tree.add_state(terminal_state, **{'init': False, 'accepting': False})

        self.add_edges(self.construct_tree(terminal_state_name=terminal_state))
        stop = time.time()
        print(f"Time to construct the Admissbility Tree: {stop - start:.2f}")
        # sys.exit(-1)
        
        # finally, run the modified VI algorithm
        reachability_game_handle = HopefulPermissiveValueIteration(game=self._adm_tree, competitive=True)
        reachability_game_handle.solve(debug=False, plot=True, extract_strategy=True)

        self._sys_best_effort_str = reachability_game_handle.sys_str_dict
        self._env_best_effort_str = reachability_game_handle.env_str_dict
        self._best_effort_state_values = reachability_game_handle.state_value_dict

        # sanity checker to check if strategies are history dependent or not.
        # tree_to_org_game_str = {}
        # for tree_s in _adm_tree._graph.nodes:
        #     if tree_s != terminal_state and _adm_tree.get_state_w_attribute(tree_s, "player") == "eve":
        #         org_s, _ = tree_s    
        #         # if state already exists, enforce its the same strategy
        #         if org_s in tree_to_org_game_str:
        #             assert tree_to_org_game_str[org_s] == self.sys_best_effort_str[tree_s], \
        #             f"[Error] Strategies are history dependent! Need to investigate this! Got two different strategies {tree_to_org_game_str[org_s]} and {self.sys_best_effort_str[tree_s]} "
        #         else:
        #             tree_to_org_game_str[org_s] = self.sys_best_effort_str[tree_s][0]


        # if plot:
            # self.add_str_flag(adm_tree=graph_tree)
            # graph_tree.plot_graph()
            # self.game.plot_graph()


class QuantitativeNaiveAdmissible(QuantitativeHopefullAdmissibleReachSyn):
    """
     Overrides baseclass to construct a tree up until a pyaoff is reached. 
    """

    def __init__(self, budget: int,  game: TwoPlayerGraph, debug: bool = False) -> 'QuantitativeHopefullAdmissibleReachSyn':
        super().__init__(game, debug)
        self._sys_best_effort_str = defaultdict(lambda: set({}))  # overide base class instation as in the base class the dictionary was returned from VI code.
        self._budget = budget
        self._adv_coop_state_values: Dict[Tuple, int] = defaultdict(lambda: math.inf)
        self._adm_tree.graph_name = "adm_tree_budget"

    @property
    def budget(self):
        return self._budget
    
    @property
    def adv_coop_state_values(self):
        return self._adv_coop_state_values

    @budget.setter
    def budget(self, budget: Union[int, float]):
        if isinstance(budget, float):
            warnings.warn("[Warning] Budget entered is a float. Floor-ing the number")
        
        self._budget = math.floor(budget)
    
    def add_str_flag(self):
        # super(AbstractBestEffortReachSyn, self).add_str_flag()
        self.game.set_edge_attribute('strategy', False)

        for curr_node, next_node in self._sys_best_effort_str.items():
            if isinstance(next_node, set) or isinstance(next_node, list):
                for n_node in next_node:
                    self.game._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.game._graph.edges[curr_node, next_node, 0]['strategy'] = True
    

    def add_edges(self, ebunch_to_add, **attr) -> None:
        """
         A function to add all the edges in the ebunch_to_add. 
        """
        init_node_added: bool = False
        for e in ebunch_to_add:
            # u and v as 2-Tuple with node and node attributes of source (u) and destination node (v)
            u, v, dd = e
            u_node = u[0]
            u_attr = u[1]
            v_node = v[0]
            v_attr = v[1]
           
            key = None
            ddd = {}
            ddd.update(attr)
            ddd.update(dd)

            # add node attributes too
            self._adm_tree._graph.add_node(u_node, **u_attr)
            self._adm_tree._graph.add_node(v_node, **v_attr)

            if u_attr.get('init') and not init_node_added:
                init_node_added = True
            
            if init_node_added and 'init' in u_attr:
                del u_attr['init']
            
            # add edge attributes too 
            if not self._adm_tree._graph.has_edge(u_node, v_node):
                key = self._adm_tree._graph.add_edge(u_node, v_node, key)
                self._adm_tree._graph[u_node][v_node][key].update(ddd)
    

    def construct_tree(self, terminal_state_name: str = "vT") -> Generator[Tuple, None, None]:
        """
         This method override the base methods. It constructs a tree in a non-recurisve depth first fashion for all plays in the original graph whose payoff <+ budger.
        """
        source = self.game_init_states[0][0]
        nodes = [source]

        visited = set()
        for start in nodes:
            if start in visited:
                continue

            visited.add(start)
            # node, payoff, # of occ, child node
            stack = [(start, 0, 1, iter(self.game._graph[start]))] 
            stack_tree_state: Dict[Tuple[str, int], int] = {(start, 0) : 1} ## to track nodes in the tree
            stack_state_only: List[str] = [start]

            while stack:
                parent, pweight, pocc, children = stack[-1]
                for child in children:
                    cweight = pweight + self.game._graph[parent][child][0]['weight'] 
                    
                    # book keeping, did we visit this state before?
                    if stack_tree_state.get((child, cweight)) is None:
                        cocc = 1    
                    else:
                        cocc = stack_tree_state.get((child, cweight)) + 1
                    
                    stack_tree_state[(child, cweight)] = cocc
                    if child in self.target_states:
                        yield ((parent, pweight, pocc), self.game._graph.nodes[parent]), ((child, cweight, cocc), self.game._graph.nodes[child]), {'weight': cweight}
                    else:
                        if cweight <= self.budget:
                            yield ((parent, pweight, pocc), self.game._graph.nodes[parent]), ((child, cweight, cocc), self.game._graph.nodes[child]), {'weight': 0}

                    visited.add(child)
                    
                    if cweight <= self.budget and child not in self.target_states:
                        stack.append((child, cweight, cocc, iter(self.game._graph[child])))
                        stack_state_only.append(child)
                        break
                    # a state that was already in play is visited. Add edge to terminal state.
                    elif cweight > self.budget:
                        yield ((parent, pweight, pocc), self.game._graph.nodes[parent]), ((terminal_state_name), {}), {'weight': 0}

                else:
                    stack.pop()
                    stack_state_only.pop()
        
        # add self loop to terminal state
        yield ((terminal_state_name), {}), ((terminal_state_name), {}), {'weight': 0}
    

    def compute_adversarial_cooperative_value(self):
        """
         A function that compute the adversarial-cooperative value for each state in the unrolled graph
        """
        for curr_node in self.game._graph.nodes():
            if not self.game._graph.nodes(data='accepting')[curr_node]:
                adv_val: int = self.winning_state_values[curr_node]
                coop_succ_vals: List[Tuple[str, int]] = []
                for succ_node in self.game._graph.successors(curr_node):
                    if self.winning_state_values[succ_node] <= adv_val:
                        coop_succ_vals.append((succ_node, self.coop_winning_state_values[succ_node]))
                
                _, min_val = min(coop_succ_vals, key=operator.itemgetter(1))
        
                self._adv_coop_state_values[curr_node] = min_val
            
            # acVal for accepting states in zero
            else:
                self._adv_coop_state_values[curr_node] = 0
    

    def check_admissible_edge(self, source: Tuple[str, int, int], succ: Tuple[str, int, int], avalues: Set[int]) -> bool:
        """
         A function that check if the an edge is admissible or not when traversing the tree of plays.
        """
        if self.coop_winning_state_values[succ] < min(avalues):
            return True
        elif self.winning_state_values[source] == self.winning_state_values[succ] == self.coop_winning_state_values[succ] == self.adv_coop_state_values[source]:
            return True
        
        return False
    

    def compute_best_effort_strategies(self, plot: bool = False):
        """
         In this algorithm we call the modified Value Iteration algorithm to computer permissive Admissible strategies.

         The algorithm is as follows:
            First, we create a tree of bounded depth (u <= budget)
            Finally, we return the strategies. 
        """
        # now construct tree
        start = time.time()
        terminal_state: str = "vT"
        self._adm_tree.add_state(terminal_state, **{'init': False, 'accepting': False})

        self.add_edges(self.construct_tree(terminal_state_name=terminal_state))
        stop = time.time()
        print(f"Time to construct the Admissbility Tree: {stop - start:.2f}")

        # compute aVal and cVal
        self._game = self._adm_tree
        # manually add the terminal state to Env player
        self.game.add_state_attribute(terminal_state, "player", "adam")  

        # get winning strategies
        self.compute_winning_strategies(permissive=True, plot=False)

        # get cooperative winning strategies
        self.compute_cooperative_winning_strategy(plot=False)

        # compute acVal for each state
        self.compute_adversarial_cooperative_value()
        
        # Admissibility setup
        # returns an iterator
        def states_from_iter(node):
            return iter(self.game._graph[node])

        visited = set()

        # normal DFS in preorder traversal
        source = self.game.get_initial_states()[0][0]
        visited.add(source)
        stack = [(source, states_from_iter(source))]
        avalues = [self.winning_state_values[source]]  # set of adversarial values encountered up until now.

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                # if child in visited:
                #     print(f"{parent}: {child}")
                # else:
                visited.add(child)
                # print(f"{parent}: {child}")
                
                # only append if the edge from parent to child is admissible and update avalues too
                if self.game._graph.nodes(data='player')[parent] == 'eve' and self.check_admissible_edge(source=parent, succ=child, avalues=avalues):
                    stack.append((child, states_from_iter(child)))
                    print(f"Admissible: {parent}: {child}")
                    self._sys_best_effort_str[parent] = self._sys_best_effort_str[parent].union(set([child])) 
                    avalues.append(self.winning_state_values[child])
                else:
                    assert self.game._graph.nodes(data='player')[parent] in['adam', 'eve'], f"[Warning] Encountered state: {parent} without player attribute in the tree."
                    if parent != 'vT':
                        stack.append((child, states_from_iter(child)))
                        print(f"Env state: {parent}: {child}")
                        avalues.append(self.winning_state_values[child])
            
            # when you come across leaf node, stop exploring
            except StopIteration:
                stack.pop()
                avalues.pop()
        
        # start DFS
        # for start_node in nodes:
        #     # history = [(start_node, set({self.winning_state_values[start_node]}))]
        #     history = [start]
        #     while history:
        #         current_node, adv_val = history[-1]
        #         if current_node not in visited_nodes:
        #             states[current_node] = states_from(current_node)
        #             # check for admissibility
        #             for succ_node in states[current_node]:
        #                 if self.coop_winning_state_values[succ_node] < min(adv_val):
        #                     admissible_edges.append((current_node, succ_node))
        #                 elif self.winning_state_values[start_node] == self.winning_state_values[succ_node] == self.coop_winning_state_values[succ_node] == self.adv_coop_state_values[current_node]:
        #                     admissible_edges.append((current_node, succ_node))
                        
        #                 history.append((succ_node, self.winning_state_values[current_node]))

        #             visited_nodes.add(current_node)
                    

        #         if len(states[current_node]) == 0:
        #         # except StopIteration:
        #             # No more edges from the current node.
        #             history.pop()
                
        #         # add condition to pop node if state is vT
        #         elif current_node[0] == 'vT':
        #             history.pop()

                # else:
                    # if state not in visited_nodes:
                        # visited_nodes.add(state)
                        # Mark the traversed "to" node as to-be-explored.
                        # if check_reverse and edge[-1] == REVERSE:
                        #     stack.append(edge[0])
                        # else:
                    # history.append((current_node, self.winning_state_values[current_node]))


        if plot:
            self.add_str_flag()
            print(f"No. of nodes in the Tree :{len(self.adm_tree._graph.nodes())}")
            print(f"No. of edges in the Tree :{len(self.adm_tree._graph.edges())}")
            self.adm_tree.plot_graph(alias=False)

