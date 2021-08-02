import copy
import queue
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Union, Optional

from ..graph import TwoPlayerGraph
from .value_iteration import ValueIteration
from .adversarial_game import ReachabilityGame


class WeightArray:
    def __init__(self, array):
        self.array = np.array(array)

    # def __eq__(self, other) -> bool:
    #     return self.array == other.array

    # def __ge__(self, other) -> bool:
    #     # Check for all elements in the array
    #     n_dim = len(self.array)
    #     for i in range(n_dim):
    #         if self.array[i] < other.array[i]:
    #             return False

    #     return True

    # def __le__(self, other) -> bool:
    #     # Check for all elements in the array
    #     n_dim = len(self.array)
    #     for i in range(n_dim):
    #         if self.array[i] > other.array[i]:
    #             return False

    #     return True


# TODO: Deal with single values rather than lists
class MinimumElementSet:

    _weight_dim = 0
    _array = None

    def __init__(self, array: List = None):
        """
        :args array:        Multi dimensional array
        """
        # If 1D, simply make it 2D
        array = np.array(array)
        if array.ndim == 1:
            array = np.array([array])

        self._weight_dim = self._sanity_check(array)

        if self._weight_dim:
            self._array = array

    @property
    def max(self):
        return len(self._array)

    def _sanity_check(self, array: np.ndarray):
        """
        Checks if input array is a valid 2D array
        with same dimension size

        :args array:        Multi dimensional array
        """
        if array is None or len(array) == 0:
            return 0

        # If 3D, 4D or ND array, raise an error
        if array.ndim > 2:
            raise ValueError('An array has to be 1D or 2D array')

        weight_dim = len(array[0])
        for arr in array:
            if len(arr) != weight_dim:
                raise ValueError('All elements should have the same size of dimensions')

        return weight_dim

    def add(self, element_list: Union[int, float, List, np.ndarray]):
        """
        When we add an element (a list) to an array,
        it has to be a minimum element.

        Definition of a Minimum Element
            At node s, let two multi-dimensional weights be e and e'.
            We define minimum element as
                (s,e) <= (s', e') iff s=s' and e <= e'

        In other word, weight e has to be less than or equal to e' for all dimensions.
        """
        if isinstance(element_list, (int, float)):
            element_list = np.array([element_list])

        if not isinstance(element_list, (List, np.ndarray)):
            raise TypeError('Element has to be a list or np.array')

        if isinstance(element_list, List):
            element_list = np.array(element_list)

        # If the current array is empty, initialize it with the element
        if self._weight_dim == 0:
            self._array = np.array([element_list])
            return

        # Checks if the new element has same dimension as others
        if len(element_list) != self._weight_dim:
            msg = f'An element of a set must be a size of {self._weight_dim}'
            raise ValueError(msg)

        # If an array already exists, check if it is a minimum element
        if self._check_if_minimum(element_list):
            # Remove all elements that are larger than the minimum_element
            delete_indices = []
            for i, arr in enumerate(self._array):
                if all(element_list <= arr):
                    # deletethe element
                    delete_indices.append(i)

            self._array = np.append(self._array, [element_list], axis=0)
            self._array = np.delete(self._array, delete_indices, axis=0)

    def _check_if_minimum(self, element_list: np.ndarray):
        # TODO: Without using for loops
        for arr in self._array:
            if all(element_list >= arr):
                # Mark it as not minimum
                return False

        return True

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.max:
            result = self._array[self._i]
            self._i += 1
            return result
        else:
            raise StopIteration

    def __eq__(self, others):
        return np.array_equal(self._array, others._array)

    def __str__(self):
        return str(self._array)


class MultiObjectiveSolver:
    """
    Currently, only supported for two weights, weight and auto_weight.
    TODO: Support for multiple objective functions

    :attribute _Wg_graph: Graph with Game Weights
    :attribute _Wa_graph: Graph with Automaton Weights
    """
    def __init__(self, graph: TwoPlayerGraph):
        """
        """
        self._game = graph
        self._Wg_graph = graph
        self._Wa_graph = self.__replace_weights_with_auto_weights()
        self._pareto_points = {}

    def __replace_weights_with_auto_weights(self) -> TwoPlayerGraph:
        graph = copy.deepcopy(self._Wg_graph)

        for data in graph._graph.edges.data():
            u_node = data[0]
            v_node = data[1]
            attributes = data[2]
            weight = attributes['weight']
            auto_weight = attributes['weights']['pref']

            graph._graph[u_node][v_node][0]['weight'] = auto_weight

        return graph

    def solve(self, minigrid_instance = None,
                    cooperative: bool = False,
                    # cooperative: bool = True,
                    epsilon: float = 0,
                    integrate_accepting: bool = False,
                    simulate_minigrid: bool = False,
                    plot: bool = False,
                    debug: bool = True):

        # # # # compute cooperative str
        # if cooperative:
        #     autoVI = ValueIteration(self._Wa_graph, competitive=False, int_val=False)
        #     autoVI.cooperative_solver(debug=True, plot=plot)
        #     gameVI = ValueIteration(self._Wg_graph, competitive=False, int_val=False)
        #     gameVI.cooperative_solver(debug=True, plot=plot)
        # # compute competitive values from each state
        # else:
        #     autoVI = ValueIteration(self._Wa_graph, competitive=True, int_val=False)
        #     autoVI.solve(debug=True, plot=plot)
        #     gameVI = ValueIteration(self._Wg_graph, competitive=True, int_val=False)
        #     gameVI.solve(debug=True, plot=plot)
        # min_Was = self.__compute_all_Wa(autoVI, gameVI)

        pareto_points = self._compute_pareto_points()
        print('Pareto Points\n', pareto_points)

        # Key: (Wa, Wg), Value: Strategy
        # for min_Wa in min_Was:
        #     Wa, Wg, strategy = self.__find_strategy_with_auto_bound(autoVI, gameVI, min_Wa)
        #     self._pareto_points[(Wa, Wg)] = strategy

        # Plot Pareto Front x=Wa, y=Wg
        # if plot:
        #     self.__plot_pareto()

    def _compute_pareto_points(self, debug=True):
        # 1. Compute state values
        reachability_game_handle = ReachabilityGame(game=self._game)
        reachability_game_handle.reachability_solver()

        accept_node = self._Wa_graph.get_accepting_states()[0]

        # 2. Compute All possible solutions' weights
        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        n_weight = len(weights)
        # Inside set([]), it had to be unhashable object so I implemented WeightArray.
        weights_set = defaultdict(lambda: MinimumElementSet(np.inf*np.ones(n_weight)))
        weights_set[accept_node] = MinimumElementSet(np.zeros(n_weight))
        weights_set_prev = None

        iter_var = 0

        def _converged(curr: Dict, prev: Dict) -> bool:
            if curr is None or prev is None:
                return False
            curr_keys = list(curr.keys())
            prev_keys = list(prev.keys())
            if curr_keys != prev_keys:
                return False
            for key in curr_keys:
                if curr[key] != prev[key]:
                    return False
            return True

        while not _converged(weights_set, weights_set_prev):

            if debug:
                print(f"{iter_var} Iterations")

            weights_set_prev = copy.deepcopy(weights_set)
            iter_var += 1

            for u_node in reachability_game_handle.sys_winning_region:

                player = self._game.get_state_w_attribute(u_node, "player")

                for v_node in self._game._graph.successors(u_node):
                    # Only allow transitions to the winning region
                    if v_node not in reachability_game_handle.sys_winning_region:
                        continue
                    # Avoid self loop
                    if u_node == v_node:
                        continue

                    # get u to v's edge weight
                    edge_weight = self._game.get_edge_weight(u_node, v_node)
                    edge_weights = self._game.get_edge_attributes(u_node, v_node, 'weights')

                    # compute u's weight (u -- edge --> v)
                    # TODO: For Env Player, play adversarial
                    for weight_array in weights_set[v_node]:
                        u_node_weight = np.array(list(edge_weights.values())) + weight_array
                        weights_set[u_node].add(u_node_weight)

        self.weights_set = weights_set
        n_init = self._game.get_initial_states()[0][0]

        return self.weights_set[n_init]

    def __compute_all_Wa(self, autoVI: ValueIteration, gameVI: ValueIteration,
                         debug: bool =True) -> Set[float]:
        """
        Save possible values as a list at each state
        Continue until it converges

        Get the list at the initial state == all possible solutions
        """
        init_node = self._Wa_graph.get_initial_states()[0][0]
        min_auto_cost = autoVI.state_value_dict[init_node]
        min_game_cost = gameVI.state_value_dict[init_node]

        # 1. Get the winning strategy for Wg, and run on Wa to get max_auto_weight
        curr_node = init_node
        accept_node = self._Wa_graph.get_accepting_states()[0]
        max_auto_weight = 0
        trace = [curr_node]
        while curr_node != accept_node:
            # Use game VI's strategy to compute the cost on Wa
            next_node = gameVI.str_dict[curr_node]
            # Get the weight from Wa
            weight = self._Wa_graph.get_edge_attributes(curr_node, next_node, 'weight')

            max_auto_weight += weight
            curr_node = next_node
            trace.append(curr_node)

        # 2. Find all possible path weight within the bound of max_auto_weight
        # Initialize a weight list at each node
        weights_set = copy.deepcopy(autoVI.state_value_dict)
        weights_set_prev = None

        for n in self._Wa_graph._graph.nodes():
            weights_set[n] = set([weights_set[n]])

        iter_var = 0

        while not weights_set == weights_set_prev:

            if debug:
                print(f"{iter_var} Iterations")

            weights_set_prev = copy.deepcopy(weights_set)
            iter_var += 1

            for u_node in self._Wa_graph._graph.nodes():

                player = self._Wa_graph.get_state_w_attribute(u_node, "player")
                if player == 'adam':
                    continue

                for v_node in self._Wa_graph._graph.successors(u_node):

                    # get u to v's edge weight
                    edge_weight = self._Wa_graph.get_edge_weight(u_node, v_node)

                    # compute u's weight (u -- edge --> v)
                    u_node_weights = set()
                    for w in weights_set[v_node]:
                        u_node_weight = edge_weight + w
                        if u_node_weight <= max_auto_weight:
                             u_node_weights.update([u_node_weight])

                    # Add to the existing set
                    weights_set[u_node].update(u_node_weights)

        self.weights_set = weights_set
        n_init = self._Wa_graph.get_initial_states()[0][0]

        return self.weights_set[n_init]

    def __find_strategy_with_auto_bound(self, autoVI: ValueIteration, gameVI: ValueIteration,
                                        Wa_bound: float) -> Tuple[float, float, Dict]:
        # 1 Given the bound, find all invalid transitions
        Wa_graph = copy.deepcopy(self._Wa_graph)
        Wg_graph = copy.deepcopy(self._Wg_graph)

        init_node = Wa_graph.get_initial_states()[0][0]
        state_value_dict = autoVI.state_value_dict

        search_queue = queue.Queue()
        visited = defaultdict(lambda: False)

        search_queue.put(init_node)
        visited[init_node] = True

        while not search_queue.empty():
            curr_node = search_queue.get()

            for next_node in Wa_graph._graph.successors(curr_node):
                next_state_value = state_value_dict[next_node]
                edge_weight = Wa_graph.get_edge_weight(curr_node, next_node)

                curr_transition_value = next_state_value + edge_weight

                # Label invalid if the transition value is larger than the bound
                if curr_transition_value > Wa_bound:
                    Wa_graph._graph[curr_node][next_node][0]['valid'] = False
                    Wg_graph._graph[curr_node][next_node][0]['valid'] = False
                else:
                    Wa_graph._graph[curr_node][next_node][0]['valid'] = True
                    Wg_graph._graph[curr_node][next_node][0]['valid'] = True

                if not visited[next_node]:
                    search_queue.put(next_node)
                    visited[next_node] = True

        # 2 Find a shortest path that minimizes the other weight within the valid traces.
        # value iterate over Wg graph
        state_value_dict = gameVI.state_value_dict

        curr_node = init_node
        accept_node = Wg_graph.get_accepting_states()[0]
        trace = [curr_node]
        wg_weight = 0
        while curr_node != accept_node:
            # Get current strategy
            # Check if it's valid, then this is the min strategy
            next_node = gameVI.str_dict[curr_node]

            # Otherwise find a tansition that is valide and minimizes the cost
            if not Wg_graph.get_edge_attributes(curr_node, next_node, 'valid'):
                next_node = self.__compute_next_transition(Wg_graph, state_value_dict, curr_node)
                if next_node is None:
                    return Wa_bound, None, []

            weight = Wg_graph.get_edge_attributes(curr_node, next_node, 'weight')

            curr_node = next_node
            trace.append(curr_node)
            wg_weight += weight

        return Wa_bound, wg_weight, trace

    def __compute_next_transition(self, Wg_graph, state_value_dict, curr_node):
        # Find valid transitions

        min_weight = np.inf
        max_weight = 0

        min_node = None
        max_node = None

        for next_node in Wg_graph._graph.successors(curr_node):
            if not Wg_graph.get_edge_attributes(curr_node, next_node, 'valid'):
                continue

            edge_wieght = Wg_graph.get_edge_attributes(curr_node, next_node, 'weight')
            next_node_weight = state_value_dict[n]
            curr_node_weight = next_node_weight + edge_wieght

            if curr_node_weight < min_weight:
                min_weight = curr_node_weight
                min_node = next_node

            if curr_node_weight > max_weight:
                max_weight = curr_node_weight
                max_node = next_node

        player = Wg_graph.get_state_w_attribute(curr_node, "player")
        if player == 'eve':
            return min_node
        else:
            return max_node

    def __plot_pareto(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for (Wa, Wg) in self._pareto_points.keys():
            ax.plot(Wa, Wg, 'x')

        ax.set_xlabel('Wa')
        ax.set_ylabel('Wg')
        plt.show()

    # TODO: Also accept pareto point's number for easy selection
    def select(self, Wa: float, Wg: float, animate=True) -> None:
        # Check if (Wa, Wg) exist
        if (Wa, Wg) not in self._pareto_points:
            raise ValueError(f'{(Wa, Wg)} is not in the dictionary')
        strategy = self._pareto_points[(Wa, Wg)]

        # TODO:
        # 1. Graph
        # 2. Animate on the sys
