import sys
import math
import copy
import time
import warnings
import numpy as np
import networkx as nx

from networkx import DiGraph
from typing import List, Dict, Tuple, Set

from numpy import ndarray
from collections import defaultdict 

from ..helper_methods import timer_decorator
from ..graph import TwoPlayerGraph
from ..graph import graph_factory
from .value_iteration import ValueIteration


# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class TopologicalValueIteration(ValueIteration):
    """
     A class that overrides the ValueIteration class to compute topologically's informed value iteration
    """
    def __init__(self, game, competitive: bool = False, int_val: bool = True, condensed_graph: DiGraph = None, scc_order: List[int] = None, sanity_check: bool = True):
        super().__init__(game, competitive, int_val)
        self._states_by_payoff = None
        self._valid_payoff_values = None
        self.condensed_graph = condensed_graph
        self.scc_order = scc_order
        if condensed_graph is None:
            self.condensed_graph, self.scc_order = self.get_nx_kosaraju_sort()
        if sanity_check:
            self._check_scc_has_all_states()
            self._check_trap_topological_order()
            
            # SCC related stuff
            # adjacency_list = self.get_adjacency_list()
            # sccs = StronglyConnectedComponentComputation(adjacency_list).get_result()
            
            # # debugging
            # print("Ordering according to Tarjan's algorithm")
            # for iter, scc in enumerate(sccs):
            #     print(f"Order: {iter}")
            #     print(', '.join([str(self.node_int_map.inverse[s]) for s in scc]))

        # if 'utls' in game.graph_name:
        #     self._get_nodes_per_gou()
        # elif 'alts' in game.graph_name:
        #     self._get_nodes_per_alt()
        # else:
        #     warnings.warn(f"[Waring] Graph name {game.graph_name} is not recognized.]")
        #     sys.exit(1)
        
        # if experimental:
        #     print("Sorted layers")
        #     for u in self.valid_payoff_values:
        #         print(f"Val: {u}")
        #         print(', '.join([str(s) for s in self.states_by_payoff[u]['nodes']]))
        #     sys.exit(-1)

    

    @property
    def states_by_payoff(self):
        return self._states_by_payoff
    
    @property
    def valid_payoff_values(self):
        return self._valid_payoff_values

    def _get_nodes_per_gou(self):
        """
         A helper function that computes the number of states for each valid Payoff value in the Graph of Utility (GoU)
        """
        if self._states_by_payoff is not None:
            return self._states_by_payoff
        
        states_by_payoff = defaultdict(lambda : {'nodes': set(), 'int_nodes': set()})
        max_payoff = 0

        for gou_state, state_data in self.org_graph.get_states():
            if gou_state == 'vT':
                continue
            _, u = gou_state[0], gou_state[1]
            if u > max_payoff:
                max_payoff = u 
            assert gou_state not in states_by_payoff[u]['nodes'], f"[Error] State {gou_state} is already in the set of states with payoff {u}."
            states_by_payoff[u]['nodes'].add(gou_state)
            states_by_payoff[u]['int_nodes'].add(self.node_int_map[gou_state])
        
        # manually add the terminal state to the set of states with highest payoff
        states_by_payoff[max_payoff]['nodes'].add('vT')
        states_by_payoff[max_payoff]['int_nodes'].add(self.node_int_map['vT'])

        self._valid_payoff_values = sorted(list(states_by_payoff.keys()), reverse=True)
        self._states_by_payoff = states_by_payoff
    

    def _check_scc_has_all_states(self) -> None:
        """
         A method that checks if the SCC has all the states in the graph.
        """
        set_of_states = set()
        for scc_state, org_state_set in self.condensed_graph.nodes.data():
            set_of_states = set_of_states.union(org_state_set['members'])

        try:
            assert set(self.org_graph._graph.nodes) == set_of_states, "[Error] The SCC does not have all the states in the graph."
        except AssertionError:
            print(f"The states are missing from the SCCs are: {self.org_graph._graph.nodes - set_of_states}")
    
    def _check_trap_topological_order(self) -> None:
        """
         A method that check if trap state is the first or second state in the topological order. 
         
         Note: This is true only when we construct the DFA games with absorbing flag set to True
        """
        first_scc, second_scc = self.condensed_graph.nodes[0]['members'], self.condensed_graph.nodes[1]['members']
        if len(self.org_graph.get_trap_states()) > 0:
            assert len(self.org_graph.get_trap_states()) == 1, "[Error] More than one trap state is present in the graph."
            if not(self.org_graph.get_trap_states()[0] in first_scc or self.org_graph.get_trap_states()[0] in second_scc):
                raise AssertionError("[Error] The trap state is not in the first or second SCC.")

    
    @timer_decorator
    def get_nx_kosaraju_sort(self) -> Tuple[DiGraph, List[int]]:
        # import networkx as nx
        # start = time.time()
        # sccs = nx.strongly_connected_components(self.org_graph._graph)
        # for scc in sccs:
        #     # print(type(scc))
        #     print(scc)
        # stop = time.time()
        # print(f"******************** SCC Computation: {stop - start} seconds ********************")

        start = time.time()
        condensed_graph = nx.condensation(self.org_graph._graph)
        stop = time.time()
        print(f"******************** Condensed Graph: {stop - start} seconds ********************")
        # print(condensed_graph.nodes)
        # print("SCC order in which the values iteration code shold run.")
        scc_order = list(reversed(list(nx.topological_sort(condensed_graph))))


        # self._adm_tree: TwoPlayerGraph = graph_factory.get("TwoPlayerGraph",
                                                        #    graph_name="SCC_DAG_NO_UNREACHABLE",
                                                        #    config_yaml="config/SCC_DAGNO_UNREACHABLE",
                                                        #    save_flag=True,
                                                        #    from_file=False, 
                                                        #    plot=False)
        # self._adm_tree._graph = condensed_graph
        # self._adm_tree.plot_graph()
        print(f"{max(scc_order) + 1} - Number of SCCs")

        return condensed_graph, scc_order

        # sys.exit(-1)
    

    def _get_nodes_per_alt(self):
        """
         A helper function that computes the number of states for each valid Payoff value in the Graph of Alternative (GoAlt)
        """
        if self._states_by_payoff is not None:
            return self._states_by_payoff
        
        states_by_payoff = defaultdict(lambda : {'nodes': set(), 'int_nodes': set()})
        max_payoff = 0
        set_of_terminal_states = set()

        for goalt_state, state_data in self.org_graph.get_states():
            # vT states dont' have DFA states associated with them. 
            if isinstance(goalt_state[0], tuple):
                org_state, u = goalt_state[0][0], goalt_state[0][1]
            elif isinstance(goalt_state, tuple) and len(goalt_state) == 2:
                org_state, u = goalt_state[0], goalt_state[1]
            else:
                warnings.warn(f"[Error] State {goalt_state} is not recognized.")
                sys.exit(1)

            if 'vT' in org_state:
                set_of_terminal_states.add(goalt_state)
                continue
            assert isinstance(u, int), f"[Error] Payoff value is not an integer. Got unsupported type{u}]"
            if u > max_payoff:
                max_payoff = u
            assert goalt_state not in states_by_payoff[u]['nodes'], f"[Error] State {goalt_state} is already in the set of states with payoff {u}."
            states_by_payoff[u]['nodes'].add(goalt_state)
            states_by_payoff[u]['int_nodes'].add(self.node_int_map[goalt_state])
        
        # manually add the terminal state to the set of states with highest payoff
        states_by_payoff[max_payoff]['nodes'].union('set_of_terminal_states')
        for _s in set_of_terminal_states:
            states_by_payoff[max_payoff]['int_nodes'].add(self.node_int_map[_s])
        
        self._valid_payoff_values = sorted(list(states_by_payoff.keys()), reverse=True)
        self._states_by_payoff = states_by_payoff
    

    def _check_scc_has_only_accp_or_trap_states(self, scc_num: int) -> bool:
        """
         A method that checks if the SCC has only accepting states.
        """
        # for scc_state, org_state_set in self.condensed_graph.nodes.data():
        if self.condensed_graph.nodes[scc_num]['members'] == set(self.org_graph.get_trap_states()):
            return True 
        elif self.condensed_graph.nodes[scc_num]['members'].issubset(self._accp_states):
            return True
        return False
    

    def get_adjacency_list(self) -> List[List]:
        """
         Construct a adjancecny of list from the graph.
        """
        print("Computing Adjacency List")
        adjacency_list: List[List] = defaultdict(lambda: [])
        for soure_target in self.org_graph._graph.adjacency():
            source =  soure_target[0]
            source_int = self.node_int_map[source]
            target_list: Dict[str, dict] = soure_target[1]

            assert isinstance(target_list, dict), "[Error] Error during parsing of Adjacency list. Fix this!"
            # adjacency_list[source].extend(target_list.keys())
            adjacency_list[source_int].extend([self.node_int_map[t] for t in target_list.keys()])
        
        return adjacency_list


    def _is_same_at_indices(self, array1: np.ndarray, array2: np.ndarray, scc_num: Set) -> bool:
        """
        Check if two arrays are the same only at a specific set of indices.

        :param array1: First array.
        :param array2: Second array.
        :param indices: List of indices to check.
        :return: True if the arrays are the same at the specified indices, False otherwise.
        """
        for state in self.condensed_graph.nodes[scc_num]['members']:
            index = self.node_int_map[state]
            if array1[index] != array2[index]:
                return False
        return True

    def update_state_values(self, val_vector: ndarray, topological_order: int) -> ndarray:
        """
        A method that back propagates the state values. Specfically, at the

            At Sys state: min_a(F(s, a) + W(s')) for all s'  
            At Env state: max_a(F(s, a) + W(s')) for all s' 

        Here F(s, a) is the edge weight and W(s') is the value of the successor state in previous iteration.
        We assume that the Sys player is trying to minimize the total cost it expends. 
        """

        val_pre = copy.copy(val_vector)
        for _n in self.condensed_graph.nodes[topological_order]['members']:
            if _n in self._accp_states and self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                continue
            
            _int_node = self.node_int_map[_n]
            val_vector[_int_node][0] = self._get_opt_val(_n, val_pre)

        self._val_vector = np.append(self.val_vector, val_vector, axis=1)

        return val_vector
    

    def solve(self, debug: bool = False, plot: bool = False, extract_strategy: bool = True, terminate_on_init: bool = False):
        # initially in the org val_vector the target node(s) will value 0
        _init_node = self.org_graph.get_initial_states()[0][0]
        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        vi_iter_var: int = 0
        terminated_early: bool = False

        for scc_num in self.scc_order:
            # if self._check_scc_has_only_accp_or_trap_states(scc_num):
            #     vi_iter_var += 1
            #     continue
            
            scc_iter_var: int = 0

            # create a local node int map for the states in this scc
            if debug: 
                print(f"********SCC Num: {scc_num}********")
            while True:
                if debug:
                    # if iter_var % 1000 == 0:
                    # print(f"{iter_var} Iterations & Payoff {payoff}")
                    print(f"{scc_iter_var} Iterations")
                
                _val_pre = copy.copy(_val_vector)
                scc_iter_var += 1

                # perform one step Value Iteration
                _val_vector: ndarray = self.update_state_values(val_vector=_val_vector, topological_order=scc_num)

                if self._is_same(_val_pre, _val_vector):
                    break
            
            vi_iter_var += 1

            if terminate_on_init and self._init_state in self.condensed_graph.nodes[scc_num]['members']:
                init_node_int = self.node_int_map.inverse[self._init_state]
                if INT_MIN_VAL < self.val_vector[init_node_int, -1] < INT_MAX_VAL:
                    terminated_early = True
                    print(f"Terminating While loop early: Reached init state after {vi_iter_var} iterations.")
                    break
                else:
                    print("Winning Strategy form init state does not exist. Not termination While loop early")

        if not terminate_on_init or not terminated_early:    
            assert vi_iter_var == len(self.scc_order), "[Error] The number of VI iterations does not match the number of SCCs."
        
        self._iterations_to_converge = vi_iter_var
        
        # safely convert values in the last col of val vector to ints
        if self._int_val:
            _int_val_vector = self.val_vector[:, -1].astype(int)
        else:
            _int_val_vector = self.val_vector[:, -1]

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: _int_val_vector[i] if INT_MIN_VAL <  _int_val_vector[i] < INT_MAX_VAL  else math.inf})

        # extract sys and env strategy after converging.
        if extract_strategy:
            self._sys_str_dict, self._env_str_dict = self.extract_strategy()

        self._str_dict = {**self._sys_str_dict, **self._env_str_dict}
        self._sys_winning_region = set(self._sys_str_dict.keys()) 

        if plot:
            self.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {vi_iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            # self._sanity_check()


class StronglyConnectedComponentComputation:
    def __init__(self, unweighted_graph):
        self.graph = unweighted_graph
        self.BEGIN, self.CONTINUE, self.RETURN = 0, 1, 2 # "recursion" handling

    def get_result(self):
        self.indices = dict()
        self.lowlinks = defaultdict(lambda: -1)
        self.stack_indices = dict()
        self.current_index = 0
        self.stack = []
        self.sccs = []

        for i in range(len(self.graph)):
            if i not in self.indices:
                self.visit(i)
        self.sccs.reverse()
        return self.sccs

    def visit(self, vertex):
        iter_stack = [(vertex, None, None, self.BEGIN)]
        while iter_stack:
            v, w, succ_index, state = iter_stack.pop()

            if state == self.BEGIN:
                self.current_index += 1
                self.indices[v] = self.current_index
                self.lowlinks[v] = self.current_index
                self.stack_indices[v] = len(self.stack)
                self.stack.append(v)

                iter_stack.append((v, None, 0, self.CONTINUE))
            elif state == self.CONTINUE:
                successors = self.graph[v]
                if succ_index == len(successors):
                    if self.lowlinks[v] == self.indices[v]:
                        stack_index = self.stack_indices[v]
                        scc = self.stack[stack_index:]
                        del self.stack[stack_index:]
                        for n in scc:
                            del self.stack_indices[n]
                        self.sccs.append(scc)
                else:
                    w = successors[succ_index]
                    if w not in self.indices:
                        iter_stack.append((v, w, succ_index, self.RETURN))
                        iter_stack.append((w, None, None, self.BEGIN))
                    else:
                        if w in self.stack_indices:
                            self.lowlinks[v] = min(self.lowlinks[v],
                                                   self.indices[w])
                        iter_stack.append(
                            (v, None, succ_index + 1, self.CONTINUE))
            elif state == self.RETURN:
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
                iter_stack.append((v, None, succ_index + 1, self.CONTINUE))
