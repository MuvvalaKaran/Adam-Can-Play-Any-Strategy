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



class QuantitativeNaiveAdmissible(AbstractBestEffortReachSyn):
    """
     Overrides baseclass to construct a tree up until a pyaoff is reached. 
    """

    def __init__(self, budget: int,  game: TwoPlayerGraph, debug: bool = False) -> 'QuantitativeNaiveAdmissible':
        super().__init__(game, debug)
        self._adm_tree: TwoPlayerGraph = graph_factory.get("TwoPlayerGraph",
                                                           graph_name="adm_tree_budget",
                                                           config_yaml="config/adm_tree_budget",
                                                           save_flag=True,
                                                           from_file=False, 
                                                           plot=False)
        self._sys_best_effort_str = defaultdict(lambda: set({}))  # overide base class instation as in the base class the dictionary was returned from VI code.
        self._budget = budget
        self._adv_coop_state_values: Dict[Tuple, int] = defaultdict(lambda: math.inf)

    @property
    def budget(self):
        return self._budget
    
    @property
    def adv_coop_state_values(self):
        return self._adv_coop_state_values
    
    @property
    def adm_tree(self):
        if len(self._adm_tree._graph) == 0:
            warnings.warn("[Error] Got empty Tree. Please run best-effort synthesis code to construct tree")
            sys.exit(-1)
        return self._adm_tree

    @budget.setter
    def budget(self, budget: Union[int, float]):
        if isinstance(budget, float):
            warnings.warn("[Warning] Budget entered is a float. Floor-ing the number")
        
        self._budget = math.floor(budget)
    
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
         Implement method to run the Value Iteration code
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
         A function that compute the adversarial-cooperative value for each state in the unrolled graph. We do not compute acVal for Env player state.
        """
        for curr_node in self.game._graph.nodes():
            if self.game._graph.nodes(data='player')[curr_node] == 'eve':
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
    

    def dfs_admissibility(self, construct_transducer: bool = False) -> None:
        """
         A helper function called from compute_best_effort_strategies() method to run the DFS algorithm in preorder fashion and
           add admissible edge to the strategy dictionary.
        
           Set the debug flag to true to print admissible edges.
        """
        # Admissibility setup
        def states_from_iter(node) -> Iterable[List]:
            return iter(self.game._graph[node])

        visited = set()
        admissible_nodes = set()
        source = self.game.get_initial_states()[0][0]
        visited.add(source)
        stack = [(source, states_from_iter(source))]
        avalues = [self.winning_state_values[source]]  # set of adversarial values encountered up until now.

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                visited.add(child)
                
                # only append if the edge from parent to child is admissible and update avalues too
                if self.game._graph.nodes(data='player')[parent] == 'eve' and self.check_admissible_edge(source=parent, succ=child, avalues=avalues):
                    stack.append((child, states_from_iter(child)))
                    if self.debug: 
                        print(f"Admissible edge: {parent}: {child}")
                    self._sys_best_effort_str[parent] = self._sys_best_effort_str[parent].union(set([child])) 
                    avalues.append(self.winning_state_values[child])
                    admissible_nodes.add(parent)
                    admissible_nodes.add(child)
                # the successor node is not admissible and showuld be removed
                elif self.game._graph.nodes(data='player')[parent] == 'eve' and not self.check_admissible_edge(source=parent, succ=child, avalues=avalues):
                    continue
                else:
                    assert self.game._graph.nodes(data='player')[parent] in 'adam', f"[Warning] Encountered state: {parent} without player attribute in the tree."
                    if parent != 'vT':
                        stack.append((child, states_from_iter(child)))
                        if self.debug:
                            print(f"Env state edge: {parent}: {child}")
                        avalues.append(self.winning_state_values[child])
                        admissible_nodes.add(parent)
                        admissible_nodes.add(child)
            
            # when you come across leaf node - accepting + Terminal state (default is vT), stop exploring
            except StopIteration:
                stack.pop()
                avalues.pop()
        
        # remove non-admissible edges from the tree
        if construct_transducer:
            non_admissible_nodes: set = set(self.game._graph.nodes) - admissible_nodes
            self.game._graph.remove_nodes_from(non_admissible_nodes)

    

    def compute_best_effort_strategies(self, plot: bool = False, plot_transducer: bool = False):
        """
         In this algorithm we call the modified Value Iteration algorithm to computer permissive Admissible strategies.

         The algorithm is as follows:
            First, we create a tree of bounded depth (u <= budget)
            Then, we compute aVal, cVal, acVal associated with each state. 
            We run a DFS algrorithm and check for admissibility of each edge using Thm. 3 in the paper. 
            Finally, return the strategies. 
        """
        # now construct tree
        start = time.time()
        terminal_state: str = "vT"
        self._adm_tree.add_state(terminal_state, **{'init': False, 'accepting': False})

        self.add_edges(self.construct_tree(terminal_state_name=terminal_state))
        stop = time.time()
        print(f"Time to construct the Admissbility Tree: {stop - start:.2f}")
        self._game = self._adm_tree
        
        # manually add the terminal state to Env player
        self.game.add_state_attribute(terminal_state, "player", "adam")  

        # get winning strategies
        print("Computing Winning strategy")
        self.compute_winning_strategies(permissive=True, plot=False)

        # get cooperative winning strategies
        print("Computing Cooperative Winning strategy")
        self.compute_cooperative_winning_strategy(plot=False)

        # compute acVal for each state
        print("Computing Adversarial-Cooperative strategy")
        self.compute_adversarial_cooperative_value()

        # Compute Admissible strategies
        print("Computing Admissible strategy")
        self.dfs_admissibility(construct_transducer=True)

        if plot:
            self.add_str_flag()
            print(f"No. of nodes in the Tree :{len(self.adm_tree._graph.nodes())}")
            print(f"No. of edges in the Tree :{len(self.adm_tree._graph.edges())}")
            self.adm_tree.plot_graph(alias=False)