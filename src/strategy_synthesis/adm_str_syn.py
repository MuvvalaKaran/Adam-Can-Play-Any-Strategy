import sys
import time
import math
import operator
import warnings

from copy import deepcopy
from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from typing import Optional, Union, List, Iterable, Dict, Set, Tuple, Generator

from ..graph import TwoPlayerGraph
from ..graph import graph_factory
from .safety_game import SafetyGame
from .value_iteration import ValueIteration, PermissiveValueIteration, PermissiveCoopValueIteration
from ...helper import InteractiveGraph



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
        self._sys_adm_str: Optional[dict] = None
        self._winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)
        self._coop_winning_state_values: Dict[str, float] = defaultdict(lambda: math.inf)

        self.debug: bool = debug
        self._game_init_states: List = self.game.get_initial_states()
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
    def sys_adm_str(self):
        return self._sys_adm_str
    
    @property
    def winning_state_values(self):
        return self._winning_state_values

    @property
    def coop_winning_state_values(self):
        return self._coop_winning_state_values
    
    @property
    def game_init_states(self):
        return self._game_init_states
    
    @game_init_states.setter
    def game_init_states(self, init_states):
        if len(init_states) > 1:
            warnings.warn("[Error] Got Multiple Initial states. Admissibility code works for one unique initial state only.")
            sys.exit(-1)
        self.game_init_states = init_states


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

        for curr_node, next_node in self._sys_adm_str.items():
            if isinstance(next_node, Iterable):
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
    def compute_adm_strategies(self):
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
        self._sys_adm_str = defaultdict(lambda: set({}))  # overide base class instation as in the base class the dictionary was returned from VI code.
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
            warnings.warn("[Error] Got empty Tree. Please run Admissibility synthesis code to construct tree")
            sys.exit(-1)
        return self._adm_tree

    @budget.setter
    def budget(self, budget: Union[int, float]):
        if isinstance(budget, float):
            warnings.warn("[Warning] Budget entered is a float. Floor-ing the number")
        
        self._budget = math.floor(budget)
    
    # Admissibility setup
    def states_from_iter(self, node) -> Iterable[List]:
        return iter(self.game._graph[node])

    
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
        admissible_nodes = set()
        source = self.game.get_initial_states()[0][0]
        stack = [(source, self.states_from_iter(source))]
        avalues = [self.winning_state_values[source]]  # set of adversarial values encountered up until now.

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                
                # only append if the edge from parent to child is admissible and update avalues too
                if self.game._graph.nodes(data='player')[parent] == 'eve' and self.check_admissible_edge(source=parent, succ=child, avalues=avalues):
                    stack.append((child, self.states_from_iter(child)))
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
                        stack.append((child, self.states_from_iter(child)))
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

    

    def compute_adm_strategies(self, plot: bool = False, plot_transducer: bool = False):
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



class QuantitativeGoUAdmissible(QuantitativeNaiveAdmissible):
    """
     This class overrides the base class's Tree construction code to construct a Graph of Utility. 
     Just like we did in Regret Synthesis and run synthesis algorithm on this graph
    """

    def __init__(self, budget: int, game: TwoPlayerGraph, debug: bool = False) -> QuantitativeNaiveAdmissible:
        super().__init__(budget, game, debug)
        self._transducer: TwoPlayerGraph = graph_factory.get("TwoPlayerGraph",
                                                           graph_name="adm_str_transducer",
                                                           config_yaml="config/adm_str_transducer",
                                                           save_flag=True,
                                                           from_file=False, 
                                                           plot=False)
        self._sys_adm_str = dict({})

    @property
    def transducer(self):
        if len(self._transducer._graph) == 0:
            warnings.warn("[Error] Got empty Transducer. Please run Admissibility synthesis code to construct Transducer")
            sys.exit(-1)
        return self._transducer


    def is_state_in_transducer(self, state, transducer_dict: dict) -> int:
        """
         A helper function that check state already visited or not. It return 1 if this is the first occurence else > 1
        """
        if transducer_dict.get(state) is None:
            transducer_dict[state] = 1
            return 1
        else:
            transducer_dict[state] += 1 
            return transducer_dict.get(state)

    
    def dfs_admissibility(self, construct_transducer: bool = False) -> None:
        """
         Override the base method. In this method we forward search over graph of utility and construct the transducer.
        """
        source = self.game.get_initial_states()[0][0]
        stack = deque()
        avalues = deque()
        # node, # of occ, child node
        stack.append((source, 1, self.states_from_iter(source)))
        book_keeing_str: Dict[Tuple[str, int], int] = {source : 1} ## to track nodes in the tree        
        if construct_transducer:
            self._transducer.add_state((source, 1), **{'player': 'eve', 'init': True})
        
        avalues.append(self.winning_state_values[source])  # set of adversarial values encountered up until now.
        while stack:
            parent, pocc, children = stack[-1]
            try:
                child = next(children)
                is_edge_adm: bool = self.check_admissible_edge(source=parent, succ=child, avalues=avalues)

                if self.game._graph.nodes(data='player')[parent] == 'eve' and is_edge_adm:
                    # add edge to the transducer
                    cocc = self.is_state_in_transducer(child, transducer_dict=book_keeing_str)
                    self.sys_adm_str[(parent, pocc)] = (child, cocc) 
                    if construct_transducer:
                        self._transducer.add_state((child, cocc), **self.game._graph.nodes[child])
                        self._transducer.add_edge((parent, pocc), (child, cocc))
                    stack.append((child, cocc, self.states_from_iter(child)))
                    avalues.append(self.winning_state_values[child])
                
                elif self.game._graph.nodes(data='player')[parent] == 'eve' and not is_edge_adm:
                    continue
                
                # all env player states belong to the transducer
                elif self.game._graph.nodes(data='player')[parent] == 'adam':
                    # add edge to the transducer
                    if parent != 'vT':
                        cocc = self.is_state_in_transducer(child, transducer_dict=book_keeing_str)
                        self.sys_adm_str[(parent, pocc)] = (child, cocc)
                        if construct_transducer:
                            self._transducer.add_state((child, cocc), **self.game._graph.nodes[child])
                            self._transducer.add_edge((parent, pocc), (child, cocc))
                        stack.append((child, cocc, self.states_from_iter(child)))
                        avalues.append(self.winning_state_values[child])
                else:
                    warnings.warn(f"[Error] Encountered a state {parent} without valid player attribute. Fix this!!!")
                    sys.exit(-1)

            except StopIteration:
                stack.pop()
                avalues.pop()


    def _remove_non_reachable_states(self, game):
        """
        A helper method that removes all the states that are not reachable from the initial state. This method is
        called by the edge weighted are reg solver method to trim states and reduce the size of the graph

        :param game:
        :return:
        """
        print("Starting purging nodes")
        # get the initial state
        _init_state = game.get_initial_states()[0][0]
        _org_node_set: set = set(game._graph.nodes())

        stack = deque()
        path: set = set()

        stack.append(_init_state)
        while stack:
            vertex = stack.pop()
            if vertex in path:
                continue
            path.add(vertex)
            for _neighbour in game._graph.successors(vertex):
                stack.append(_neighbour)

        _valid_states = path

        _nodes_to_be_purged = _org_node_set - _valid_states
        game._graph.remove_nodes_from(_nodes_to_be_purged)
        print("Done purging nodes")


    def construct_graph_of_utility(self, terminal_state_name: str = "vT") -> None:
        """
            A function to construct the graph of utility given a two-player turn-based game. 
            This function is similar to the regret synthesis code
        """
        source = self.game_init_states[0][0]

        # construct nodes
        for _s in self.game._graph.nodes():
            for _u in range(self.budget + 1):
                _org_state_attrs = self.game._graph.nodes[_s]
                _new_state = (_s, _u)
                self.adm_tree.add_state(_new_state, **_org_state_attrs)
                self.adm_tree._graph.nodes[_new_state]['accepting'] = False
                self.adm_tree._graph.nodes[_new_state]['init'] = False

                if _s == source and _u == 0:
                    self.adm_tree._graph.nodes[_new_state]['init'] = True
        
        # self-loop for the terminal states with edge weight zero.
        self.adm_tree.add_edge((terminal_state_name), (terminal_state_name))
        self.adm_tree._graph[(terminal_state_name)][(terminal_state_name)][0]['weight'] = 0

        # construct edges
        for _s in self.game._graph.nodes():
            for _u in range(self.budget + 1):
                _curr_state = (_s, _u)

                # get the org neighbours of the _s in the org graph
                for _org_succ in self.game._graph.successors(_s):
                    # the edge weight between these two, add _u to this edge weight to get _u'. Add this edge to G'
                    _org_edge_w: int = self.game.get_edge_attributes(_s, _org_succ, "weight")
                    _next_u = _u + _org_edge_w

                    _succ_state = (_org_succ, _next_u)

                    if not self.adm_tree._graph.has_node(_succ_state):
                        if _next_u <= self.budget:
                            warnings.warn(f"Trying to add a new node {_succ_state} to the graph of utility."
                                          f"This should not happen. Check your construction code")
                            continue

                    # if the next state is within the bounds then, add that state to the graph of utility with edge
                    # weight 0
                    if _next_u <= self.budget:
                        _org_edge_attrs = self.game._graph.edges[_s, _org_succ, 0]
                        self.adm_tree.add_edge(u=_curr_state, v=_succ_state, **_org_edge_attrs)
                        self.adm_tree._graph[_curr_state][_succ_state][0]['weight'] = 0

                    if _next_u > self.budget:
                        if not self.adm_tree._graph.has_edge(_curr_state, terminal_state_name):
                            self.adm_tree.add_edge(u=_curr_state, v=terminal_state_name, weight=0)
        
        # construct target states
        _accp_states: list = self.game.get_accepting_states()

        for _accp_s in _accp_states:
            for _u in range(self.budget + 1):
                _new_accp_s = (_accp_s, _u)

                if not self.adm_tree._graph.has_node(_new_accp_s):
                    warnings.warn(f"Trying to add a new accepting node {_new_accp_s} to the graph of best alternatives."
                                  f"This should not happen. Check your construction code")

                self.adm_tree.add_accepting_state(_new_accp_s)

                # also we need to add edge weight to target states.
                for _pre_s in self.adm_tree._graph.predecessors(_new_accp_s):
                    if _pre_s == _new_accp_s:
                        continue
                    self.adm_tree._graph[_pre_s][_new_accp_s][0]['weight'] = _u
    

    def compute_adm_strategies(self, plot: bool = False, purge_states: bool = True, plot_transducer: bool = False, compute_str: bool = True):
        """
         In this algorithm we call the modified Value Iteration algorithm to computer permissive Admissible strategies.

         The algorithm is as follows:
            First, we create a tree of bounded depth (u <= budget)
            Then, we compute aVal, cVal, acVal associated with each state. 
            We run a DFS algrorithm and check for admissibility of each edge using Thm. 3 in the paper. 
            Finally, return the strategies. 
        """
        # compute cooperative value and check if budget >= cVal(init_state)
        self.compute_cooperative_winning_strategy(plot=False)

        if self._coop_winning_state_values[self.game_init_states[0][0]] > self.budget:
            print(f"cVal(v0): {self._coop_winning_state_values[self.game_init_states[0][0]]}. No path to the Accepting states exisits.")
            return None

        # now construct tree
        start = time.time()
        terminal_state: str = "vT"
        self._adm_tree.add_state(terminal_state, **{'init': False, 'accepting': False, 'player': 'adam'})

        self.construct_graph_of_utility(terminal_state_name=terminal_state)
        # helper method to remove the state that cannot reached from the initial state of G'
        if purge_states:
            start = time.time()
            self._remove_non_reachable_states(self.adm_tree)
            stop = time.time()
            if self.debug:
                print(f"******************************Removing non-reachable states on Graph of Utility : {stop - start} ****************************")

        stop = time.time()
        print(f"Time to construct the Admissbility Tree: {stop - start:.2f}")
        self._game = self._adm_tree
        self._game_init_states = self.game.get_initial_states()
        self.target_states = set(s for s in self._game.get_accepting_states())

        if plot:
            self.game.plot_graph(alias=False)

        # print(f"No. of nodes in the Tree :{len(self.adm_tree._graph.nodes())}")
        # print(f"No. of edges in the Tree :{len(self.adm_tree._graph.edges())}")

        # get winning strategies
        print("Computing Winning strategy")
        self.compute_winning_strategies(permissive=True, plot=False)
        if self.is_winning():
            print("Winning stratgey exists")

        # get cooperative winning strategies
        print("Computing Cooperative Winning strategy")
        self.compute_cooperative_winning_strategy(plot=False)

        # compute acVal for each state
        print("Computing Adversarial-Cooperative strategy")
        self.compute_adversarial_cooperative_value()

        # construct_transuder: bool = True if plot_transducer else False
        if compute_str:
            # Compute Admissible strategies
            print("Computing Admissible strategy")
            self.dfs_admissibility(construct_transducer=True)

            if plot_transducer:
                print(f"No. of nodes in the Tree :{len(self.adm_tree._graph.nodes())}")
                print(f"No. of edges in the Tree :{len(self.adm_tree._graph.edges())}")
                self.transducer.plot_graph(alias=False)
        
        if self.debug:
            print(f"No. of nodes in the Tree :{len(self.adm_tree._graph.nodes())}")
            print(f"No. of edges in the Tree :{len(self.adm_tree._graph.edges())}")



class QuantitativeGoUAdmissibleWinning(QuantitativeGoUAdmissible):
    """
     Overrides the base class's admissibility checking method to enforce value-preserving property
    """

    def __init__(self, budget: int, game: TwoPlayerGraph, debug: bool = False) -> QuantitativeNaiveAdmissible:
        super().__init__(budget, game, debug)
    

    def check_admissible_edge(self, source: Tuple[str, int], succ: Tuple[str, int], avalues: Set[int]) -> bool:
        """
          A function that check if the an edge is admissible winning or not when traversing the tree of plays. 
          There is an addition check for value presevation property.
        """
        if self.coop_winning_state_values[succ] < min(avalues) and ((source not in self.winning_region) or (self.winning_state_values[succ] != math.inf)):
            return True
        elif self.winning_state_values[source] == self.winning_state_values[succ] == self.coop_winning_state_values[succ] == self.adv_coop_state_values[source]:
            return True
        
        return False


class QuantiativeRefinedAdmissible(AbstractBestEffortReachSyn):
    """
     This class inherits the class and implemente's the proposed algorithm of ICRA 25. We compute Adm strategies and progressive refine them as follows:
     1. If a admissible strategy exists, choose that, 
     2. If a safe-admissible strategy exists, choose that
     3. If a safe-admissible does not exists then play hopeful admissible strategy, finally, 
     4. If a winning admissible strategy exists then compute Wcoop and choose those strategies.

     Unlike our IJCAI 25's proposed algorithm, this algorithm is indepedent of the budget and does NOT require us to rollout the game.  
    """
    def __init__(self, game: TwoPlayerGraph, debug: bool = False):
        """
         Note: Coop Sys Str Dict is the Str for set of all Cooperative str - non-deferring str that always reach the goal state(s). 
         Coop Optimal Str: Set of all Cooperative Optimal str - non-deferring str that always reach the goal state(s) optimally. 
        """
        
        super().__init__(game, debug)
        self._wcoop: dict = defaultdict(lambda: set())
        self._play_hopeful_game: bool = False
        self._safe_adm_str : Dict[str, Union[str, Iterable]] = defaultdict(lambda: set())
        self._hopeful_adm_str: Dict[str, Union[str, Iterable]] = defaultdict(lambda: set())
        self._coop_optimal_sys_str: Dict[str, Union[str, Iterable]] = defaultdict(lambda: set())
        self._env_pending_region: set= set()
        self._sys_pending_region: set = set() 
        self._hopeful_game: PermissiveValueIteration = False
        self._safety_game: SafetyGame = None
    
    @property
    def wcoop(self):
        assert len(self._wcoop) != 0, "Please run the solver before accessing the WCoop strategies."
        return self._wcoop

    @property
    def play_hopeful_game(self):
        return self._play_hopeful_game

    @property
    def coop_optimal_sys_str(self):
        return self._coop_optimal_sys_str
    
    @property
    def safe_adm_str(self):
        return self._safe_adm_str
    
    @property
    def hopeful_adm_str(self):
        return self._hopeful_adm_str
    
    @property
    def env_pending_region(self):
        return self._env_pending_region

    @property
    def sys_pending_region(self):
        return self._sys_pending_region

    @property
    def hopeful_game(self):
        return self._hopeful_game
    

    @property
    def safety_game(self):
        return self._safety_game



    def get_pending_region(self, print_states: bool = False):
        """
        Overide base method to further sort states based on Sys and Env player.
         A Method that compute the set of states from which there does exists a path to the target state(s). 
        """
        assert bool(self._winning_region) is True, "Please Run the solver before accessing the Pending region."
        if not bool(self._losing_region):
            self._losing_region = self.game_states.difference(self._coop_winning_region)
        
        tmp_states = self._losing_region.union(self.winning_region)
        self._pending_region =  self.game_states.difference(tmp_states)

        if print_states:
            print("Pending Region: \n", self._pending_region)
        
        for state in self._pending_region:
            player: str = self.game.get_state_w_attribute(state, "player")
            if  player == "eve":
                self._sys_pending_region.add(state)
            elif player == "adam":
                self._env_pending_region.add(state)
            else:
                warnings.warn(f"[Error] State {state} does not have a player attribute.") 
        
        return self._pending_region 

    def compute_winning_strategies(self, plot: bool = True):
        """
         Implement method to run the Value Iteration code
        """
        reachability_game_handle = PermissiveValueIteration(game=self.game, competitive=True)
        reachability_game_handle.solve(debug=False, plot=plot, extract_strategy=True)
        self._sys_winning_str = reachability_game_handle.sys_str_dict
        self._env_winning_str = reachability_game_handle.env_str_dict
        self._winning_state_values = reachability_game_handle.state_value_dict

        # sometime an accepting may not have a winning strategy. Thus, we only store states that have an winning strategy
        _sys_states_winning_str = reachability_game_handle.sys_str_dict.keys()

        # update winning region and optimal state values
        for ws in reachability_game_handle.winning_region:
            if self.game.get_state_w_attribute(ws, 'player') == 'eve' and ws in _sys_states_winning_str:
                self._winning_region.add(ws)
            elif self.game.get_state_w_attribute(ws, 'player') == 'adam':
                self._winning_region.add(ws)
        
        if self.debug and reachability_game_handle.is_winning():
            print("There exists a Winning strategy from the Initial State")
    

    def compute_cooperative_winning_strategy(self, permissive: bool = False, plot: bool = False):
        """
         Override the base method to run the Permissive Coop Value Iteration code
        """
        coop_handle = PermissiveCoopValueIteration(game=self.game)
        coop_handle.solve(debug=False, plot=plot, extract_strategy=True)
        self._sys_coop_winning_str = coop_handle.sys_str_dict
        self._env_coop_winning_str = coop_handle.env_str_dict
        self._coop_winning_region = set(coop_handle.sys_str_dict.keys()).union(set(coop_handle.env_str_dict.keys()))
        self._coop_winning_state_values = coop_handle.state_value_dict
        self._coop_optimal_sys_str = coop_handle.sys_coop_opt_str_dict
        
        if self.debug and coop_handle.is_winning():
            print("There exists a path from the Initial State")
    
    def compute_wcoop_strategies(self):
        """
         A helper function that look through all the winning strategies and choose the one woth minimum optimal Cooperative value. 
        """
        for state, succ_state in self.sys_winning_str.items():
            coop_val = min(self.coop_winning_state_values[i] for i in succ_state)
            self._wcoop[state] = [i for i in succ_state if self.coop_winning_state_values[i] == coop_val]
    

    def helper_func_tic_tac_toe(self, hopeful_game_handle):
        """
         Check the reachable states for Tic-tac-toe under F(win) specfication
        """
        init_state = self.game.get_initial_states()[0][0]
        stack = deque()
        path: set = set()
        values_of_states_in_hopeless_game = set()

        stack.append(init_state)
        while stack:
            vertex = stack.pop()
            if vertex in path:
                # if vertex != 'q2':
                    # print(f"error: {vertex} already visited in the Tree")
                continue
            path.add(vertex)
            if self.game.get_state_w_attribute(vertex, "player") == 'eve':
                # get successors from the str
                try:
                    for _neighbour in self.sys_adm_str[vertex]:
                        stack.append(_neighbour)
                        # values_of_states_in_hopeless_game.add(hopeful_game_handle.state_value_dict.get(_neighbour))
                except KeyError:
                    continue
            elif self.game.get_state_w_attribute(vertex, "player") == 'adam':
                for _neighbour in self.game._graph.successors(vertex):
                    stack.append(_neighbour)
                    values_of_states_in_hopeless_game.add(hopeful_game_handle.state_value_dict.get(_neighbour))
            else:
                 print(f"error: {vertex} does not have a player attribute")

        # check if there is "losing state"
        for state in path:
            if self.game.get_state_w_attribute(state, 'ap') == 'lose':
                print(f"Losing state reachable: {state}")
        
        import pprint
        pprint.pp(values_of_states_in_hopeless_game)
        
    

    def compute_adm_strategies(self, plot: bool = False) -> None:
        """
         Main method that implements computation of Admissible strategies. 

         1. First play, Min-Min (Permissive Coop) and Min-Max game and compute Losing, Pending, and Winning regions.
         2. If initial state belongs to Losing region, then return the original game - all strategies are admissible.
         3. Play safety game with Pending and Winning as target states for the Sys player. 
         4. If safe strategy exists from all Sys player states in Pending Region then 
            4.1 Check if safety and Permissive Coop strategy's intersetion is empty for any Sys player states in Pending region
            4.2 If yes, then Remove Hopeless strategies for Env player and play Min-Max game on this hopeful game
            4.3 If no, then stitch strategies and return them. In winning region, play WCoop, in Pending play safe-admissible strategy
         5. stitch: adm - safe-adm - hopeful-adm and Wcoop strategy
            5.1. Remove Sys player edges that transit to Losing region - as they are never admissible.
            5.2. In pending region, Sys states that have safe-adm  will play that str else they will play hopeful-adm.
            5.3 In Winning region, Sys will play Wcoop as they are winning and admissible.
        """
        # get permissive cooperative winning strategies and optimal cooperative values 
        print("Computing Cooperative Winning strategy")
        start = time.time()
        self.compute_cooperative_winning_strategy(plot=False)
        stop = time.time()
        print(f"******************** Co-op Computation time: {stop - start} ********************")

        # break if init state belongs to losing region
        init_state = self.game.get_initial_states()[0][0]
        if init_state not in self.coop_winning_region:
            print("No path to the Accepting states exisits.")
            return None
        
        # get winning strategies
        print("Computing Winning strategy")
        start = time.time()
        self.compute_winning_strategies(plot=False)
        self.compute_wcoop_strategies()
        stop = time.time()
        print(f"******************** WCo-op Computation time: {stop - start} ********************")

        # break if winning str exists
        if self.is_winning():
            self._sys_adm_str = self.wcoop
            return None

        # Compute SAdm strategy
        # 1. play safety game
        # 2. Prune out unsafe actions, losing region and unsafe states
        # 3. Play permissive min-min game to compute cVals
        # 4. Construct SAdm - safe strategy that choose actions minimum cVal at the nexr state.
        # 5. If init state in safe game, break else compute hopeful strategies. 
        # play safety game
        print("Computing Safety strategy")
        start = time.time()
        safe_states: set = self.pending_region.union(self.winning_region)
        safety_handle = SafetyGame(game=self.game, target_states=safe_states, debug=self.debug, sanity_check=False)
        safety_handle.reachability_solver()
        
        # compute set of unsafe sys states from after playing the safety game
        all_sys_nodes: set = set()
        for i in self.game._graph.nodes():
            if self.game.get_state_w_attribute(i, 'player') == 'eve':
                all_sys_nodes.add(i)
        unsafe_states = all_sys_nodes.difference(safety_handle.sys_str.keys())
        safeadm_game: TwoPlayerGraph = deepcopy(self.game)
        safeadm_game._graph.remove_nodes_from(unsafe_states)
        
        # remove edges that are neither safe nor reachable admissible 
        sys_edges_to_rm = set()
        for curr_state, succ_state in safety_handle.sys_str.items():
            assert safeadm_game.get_state_w_attribute(curr_state, "player") == "eve", "[Error] Trying to remove unsafe edges from Eve's state."
            bad_succ: set =  set(self.game._graph.successors(curr_state)).difference(succ_state)
            assert bad_succ != succ_state, "[Error], removing all successor state(s). This should NOT happen! FIX THIS!!!"
            for bs in bad_succ:
                sys_edges_to_rm.add((curr_state, bs))
         # loop over to check if there are env state with no sucessors, if so remove them.
        safeadm_game._graph.remove_edges_from(sys_edges_to_rm)
        
        # after removing some sys states, there might exist Env states that do not transition to any Sys states. Need ot remove those too
        env_state_to_rm = set()
        for s in safeadm_game._graph.nodes():
            if safeadm_game.get_state_w_attribute(s, "player") == "adam" and len(list(safeadm_game._graph.successors(s))) == 0:
                env_state_to_rm.add(s)
        safeadm_game._graph.remove_nodes_from(env_state_to_rm)
        
        safe_adm_handle = PermissiveCoopValueIteration(game=safeadm_game)
        safe_adm_handle.solve(debug=False, plot=plot, extract_strategy=True)

        self._play_hopeful_game = True
        if self.game_init_states[0][0] not in unsafe_states and safe_adm_handle.is_winning():
            print("SAdm Strategy from Initial state exists!!!")
            self._play_hopeful_game = False
        elif self.game_init_states[0][0] in unsafe_states:
            print("SAdm Strategy from Initial state does NOT exists!!! :()")
        
        
        # Construct SAdm - safe strategy that choose actions minimum cVal at the nexr state.
        for sys_state, succ_states in safe_adm_handle.sys_str_dict.items():
            assert safeadm_game.get_state_w_attribute(sys_state, "player") == "eve", "[Error] Trying to add SAdm strategy from Eve's state."
            succ_vals = [(succ, safe_adm_handle.state_value_dict.get(succ)) for succ in succ_states]
            _, min_val = min(succ_vals, key=operator.itemgetter(1))
            # self._safe_adm_str[sys_state] = [(state, state_val) for state, state_val in succ_vals if min_val == state_val]
            self._safe_adm_str[sys_state] = [state for state, state_val in succ_vals if min_val == state_val]

        self._coop_winning_state_values = safe_adm_handle.state_value_dict
        self._safety_game = safe_adm_handle
        stop = time.time()
        print(f"******************** Safe-Admissible Computation time: {stop - start} ********************")
        
        # Stitch adm str - values from Sys winning str dict will overide the values from safe-adm str
        self._sys_adm_str = {**self._safe_adm_str, **self.wcoop}
        if not self._play_hopeful_game:
            InteractiveGraph.visualize_game(game=safeadm_game,
                                            strategy=self._sys_adm_str,
                                            value_dict=safe_adm_handle.state_value_dict,
                                            source=init_state,
                                            # source=(('sys', ((7, 5), 'right'), ((8, 2), 'right')), 'q1'),
                                            depth_limit=30)
        # InteractiveGraph.visualize_game(game=safety_handle._game,
        #                                 # strategy=self._sys_adm_str,
        #                                 # value_dict=safe_adm_handle.state_value_dict,
        #                                 source=init_state,
        #                                 # source=(('sys', ((7, 5), 'right'), ((8, 2), 'right')), 'q1'),
        #                                 depth_limit=30)

        if self._play_hopeful_game:
            print("Computing Hopeful strategy")
            hopeful_sys_state: set = set()
            for state in self.sys_pending_region:
                if safe_adm_handle.sys_str_dict.get(state, None) is None:
                    hopeful_sys_state.add(state)
            
            # create copy only if you are plotting.
            start = time.time()
            # hopeful_game: TwoPlayerGraph = deepcopy(self.game) if plot else self.game 
            hopeful_game: TwoPlayerGraph = deepcopy(self.game) # if plot else self.game 
            env_edges_to_rm = set()
            # remove hopeless edges - env winning str such that successor is in losing region
            for state, succ_state in safety_handle.env_str.items():
                if state in self.pending_region and succ_state in self.losing_region:
                    assert hopeful_game.get_state_w_attribute(state, "player") == "adam", "[Error] Removing hopeless edge(s) from Sys's state."
                    env_edges_to_rm.add((state, succ_state))
                    
            hopeful_game._graph.remove_edges_from(env_edges_to_rm)
            
            # remove non-admissble sys edges from Pending region to Losing region
            sys_edges_to_rm = set()
            for state in self.sys_pending_region:
                for succ_state in self.game._graph.successors(state):
                    if succ_state in self.losing_region:
                        sys_edges_to_rm.add((state, succ_state))
            
            hopeful_game._graph.remove_edges_from(sys_edges_to_rm)

            # play hopeful game - we could maybe speed this be using Values from previous Min-Max VI
            hope_game_handle: PermissiveValueIteration = PermissiveValueIteration(game=hopeful_game, competitive=True)
            hope_game_handle.solve(debug=False, plot=plot, extract_strategy=True)
            for s in hopeful_sys_state:
                if hope_game_handle.sys_str_dict.get(s):
                    self._hopeful_adm_str[s] = hope_game_handle.sys_str_dict[s]
                else:
                    # if aVal(v') = inf then all actions s.t Sys player stays in pending/ go to winning region
                    # are hopeful admissible
                    self._hopeful_adm_str[s] = list(hopeful_game._graph.successors(s))
            
            stop = time.time()
            print(f"******************** Hope-Admissible Computation time: {stop - start} ********************")
            self._hopeful_game = hope_game_handle

            # when hope-adm is same adm then FOR ROLLOUT purposes we can play one of the cooperative winning strategy.
            # cooperative winning strategy is non-deferring strategy and as such will ensure reaching goal state as long as Env cooperates
            self._sys_adm_str = {**self._hopeful_adm_str, **self._safe_adm_str, **self.wcoop}
        
        ### sanity check for F(win) - formula to check if robot every reaches a losing region?
        # self.helper_func_tic_tac_toe(hopeful_game_handle=hope_game_handle)
        # sys.exit(-1)

        if plot:
            self.add_str_flag()
            self.game.plot_graph()

        return None




