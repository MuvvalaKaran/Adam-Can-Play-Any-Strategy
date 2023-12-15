import sys
import math
import copy
import warnings
import numpy as np


from numpy import ndarray
from collections import defaultdict 
from .value_iteration import ValueIteration


# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class TopologicalValueIteration(ValueIteration):
    """
     A class that overrides the ValueIteration class to compute topologically's informed value iteration
    """
    def __init__(self, game, competitive: bool = False, int_val: bool = True):
        super().__init__(game, competitive, int_val)
        self._states_by_payoff = None
        self._valid_payoff_values = None
        if 'utls' in game.graph_name:
            self._get_nodes_per_gou()
        elif 'alts' in game.graph_name:
            self._get_nodes_per_alt()
        else:
            warnings.warn(f"[Waring] Graph name {game.graph_name} is not recognized.]")
            sys.exit(1)
    

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
    

    def update_state_values(self, val_vector: ndarray, topological_order: int) -> ndarray:
        """
        A method that back propagates the state values. Specfically, at the

            At Sys state: min_a(F(s, a) + W(s')) for all s'  
            At Env state: max_a(F(s, a) + W(s')) for all s' 

        Here F(s, a) is the edge weight and W(s') is the value of the successor state in previous iteration.
        We assume that the Sys player is trying to minimize the total cost it expends. 
        """

        val_pre = copy.copy(val_vector)
        for _n in self.states_by_payoff[topological_order]['nodes']:
            if _n in self._accp_states and self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                continue
            
            _int_node = self.node_int_map[_n]
            val_vector[_int_node][0] = self._get_opt_val(_n, val_pre)

        self._val_vector = np.append(self.val_vector, val_vector, axis=1)

        return val_vector
    

    def solve(self, debug: bool = False, plot: bool = False, extract_strategy: bool = True):
        # initially in the org val_vector the target node(s) will value 0
        _init_node = self.org_graph.get_initial_states()[0][0]
        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        iter_var: int = 0

        for payoff in self.valid_payoff_values:
            while True:
                if debug:
                    # if iter_var % 1000 == 0:
                    print(f"{iter_var} Iterations")
                
                u_val_pre = copy.copy(_val_vector)
                iter_var += 1

                # perform one step Value Iteration
                _val_vector: ndarray = self.update_state_values(val_vector=_val_vector, topological_order=payoff)

                if self._is_same(u_val_pre, _val_vector):
                    break
            
        self._iterations_to_converge = iter_var
        
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
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            # self._sanity_check()


