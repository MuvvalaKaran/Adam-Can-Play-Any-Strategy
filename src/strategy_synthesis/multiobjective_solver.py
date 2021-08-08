import copy
import queue
import warnings
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


class MinimumElementSet:

    _weight_dim = 0
    _array = None

    def __init__(self, array: List):
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
    def array(self):
        return self._array

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

    def _check_if_equal(self, element_list: np.ndarray):
        # TODO: Without using for loops
        for arr in self._array:
            if all(element_list == arr):
                return True
        return False

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


class MaximumElementSet:

    _weight_dim = 0
    _array = None

    def __init__(self, array: List, index_to_maximize: int = None):
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

            if index_to_maximize:
                if index_to_maximize < 0 or self._weight_dim <= index_to_maximize:
                    msg = f'index must be between 0-{self._weight_dim-1}'
                    raise ValueError(msg)

        self._index = index_to_maximize

    @property
    def array(self):
        return self._array

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
        if self._check_if_maximum(element_list):
            # Remove all elements that are larger than the minimum_element
            delete_indices = []
            if self._index:
                for i, arr in enumerate(self._array):
                    if element_list[self._index] >= arr[self._index]:
                        # deletethe element
                        delete_indices.append(i)
            else:
                for i, arr in enumerate(self._array):
                    if all(element_list >= arr):
                        # deletethe element
                        delete_indices.append(i)

            self._array = np.append(self._array, [element_list], axis=0)
            self._array = np.delete(self._array, delete_indices, axis=0)

    def _check_if_maximum(self, element_list: np.ndarray):
        if self._index is not None:
            for arr in self._array:
                if element_list[self._index] <= arr[self._index]:
                    return False
            return True

        # TODO: Without using for loops
        for arr in self._array:
            if all(element_list <= arr):
                # Mark it as not minimum
                return False

        return True

    def _check_if_equal(self, element_list: np.ndarray):
        # TODO: Without using for loops
        for arr in self._array:
            if all(element_list == arr):
                return True
        return False

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
    def __init__(self, graph: TwoPlayerGraph, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        """
        self._game = copy.deepcopy(graph)
        self._rtol = rtol
        self._atol = atol
        self._equal_nan = equal_nan

        self._reachability_solver = ReachabilityGame(game=self._game)
        self._reachability_solver.reachability_solver()
        self._game.delete_cycles(self._reachability_solver.sys_winning_region)

        self._weights_set = None
        self._strategies = None

    @property
    def strategies(self):
        if self._strategies is None:
            raise Exception("No strategies found. Please run 'solve' function first.")

        return self._strategies

    @property
    def pareto_points(self):
        if self._weights_set is None:
            raise Exception("No pareto points found. Please run 'solve' function first.")

        n_init = self._game.get_initial_states()[0][0]
        pareto_points = self._weights_set[n_init]._array

    def solve(self, minigrid_instance = None,
                    epsilon: float = 0,
                    integrate_accepting: bool = False,
                    simulate_minigrid: bool = False,
                    plot: bool = False,
                    debug: bool = False,
                    index_to_maximize: int = None):

        # Pareto Computation
        pareto_points = self._compute_pareto_points(debug, index_to_maximize)
        print('Pareto Points\n', pareto_points)

        # Pareto Visualization
        accept_node = self._game.get_accepting_states()[0]
        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        weight_names = list(weights.keys())

        self._plot_pareto(pareto_points, weight_names)

        # Strategy Synthesis
        self._strategies = self._compute_strategies(debug)

        # Print out the computed strategies
        for pp, strategy in self._strategies.items():
            print(f'Pareto Point: {pp}')
            for curr_node, actions in strategy.items():
                print(f'{curr_node}: {actions}')

    def _compute_pareto_points(self, debug: bool = True, index_to_maximize: int = None):

        #################### Winning Region Computation ####################

        accept_node = self._game.get_accepting_states()[0]

        #################### Pareto Points Computation ####################

        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        n_weight = len(weights)
        # Inside set([]), it had to be unhashable object so I implemented WeightArray.
        weights_set = {}
        for node in self._reachability_solver.sys_winning_region:
            player = self._game.get_state_w_attribute(node, "player")
            if player == 'eve': # System / Robot
                weights_set[node] = MinimumElementSet(np.inf*np.ones(n_weight))
            else:
                # weights_set[node] = MaximumElementSet(np.zeros(n_weight))
                weights_set[node] = MaximumElementSet(np.zeros(n_weight),
                                                      index_to_maximize=index_to_maximize)
        weights_set[accept_node] = MinimumElementSet(np.zeros(n_weight))
        weights_set_prev = None

        strategies_at_node = defaultdict(lambda: defaultdict(lambda: set()))

        iter_var = 0

        while not self._converged(weights_set, weights_set_prev):

            if debug:
                print(f"{iter_var} Iterations")

            weights_set_prev = copy.deepcopy(weights_set)
            iter_var += 1
            search_queue = queue.Queue()
            visited = set()
            search_queue.put(accept_node)
            visited.add(accept_node)

            while not search_queue.empty():

                v_node = search_queue.get()

                for u_node in self._game._graph.predecessors(v_node):
                    # Only allow transitions to the winning region
                    if u_node not in self._reachability_solver.sys_winning_region:
                        continue
                    # Avoid self loop
                    if u_node == v_node:
                        continue

                    if u_node not in visited:
                        visited.add(u_node)
                        search_queue.put(u_node)

                    player = self._game.get_state_w_attribute(u_node, "player")

                    # get u to v's edge weight
                    edge_weight = self._game.get_edge_weight(u_node, v_node)
                    edge_weights = self._game.get_edge_attributes(u_node, v_node, 'weights')
                    action = self._game.get_edge_attributes(u_node, v_node, 'actions')

                    # compute u's weight (u -- edge --> v)
                    for weight_array in weights_set[v_node]:
                        # Append action to the sequence of actions already taken (append to head)
                        # For each node, for each weight vector, store a list of actions
                        u_node_weight = np.array(list(edge_weights.values())) + weight_array
                        if u_node == ('t2', 'q1'):
                            print(u_node_weight, v_node)
                        weights_set[u_node].add(u_node_weight)

        self._weights_set = weights_set
        n_init = self._game.get_initial_states()[0][0]
        pareto_points = self._weights_set[n_init]._array

        return pareto_points

    def _compute_strategies(self, debug: bool = False):
        """
        """
        if self._weights_set is None:
            raise Exception('Pareto Points are not computed')

        n_init = self._game.get_initial_states()[0][0]
        pareto_points = self._weights_set[n_init]._array

        accept_node = self._game.get_accepting_states()[0]
        strategies = defaultdict(lambda: defaultdict(lambda: set()))

        # For each initial cumulative weight (pareto point),
        # we store paths (actions to take at each node)
        for pareto_point in pareto_points:

            curr_node = copy.deepcopy(n_init)
            current_remain_weights = copy.deepcopy(pareto_point)

            search_queue = queue.Queue()
            search_queue.put((curr_node, current_remain_weights))

            # Until we find all paths for the current pareto point
            while curr_node != accept_node and not search_queue.empty():
            # while not search_queue.empty:
                curr_node, current_remain_weights = search_queue.get()
                current_remain_weights = self._approximate_zero_weights(current_remain_weights)

                # Check for all winning nodes
                for next_node in self._game._graph.successors(curr_node):
                    if next_node not in self._reachability_solver.sys_winning_region:
                        continue

                    # Get all attributes
                    action = self._game.get_edge_attributes(curr_node, next_node, 'actions')
                    weights = self._game.get_edge_attributes(curr_node, next_node, 'weights')
                    weights = np.array(list(weights.values()))

                    next_remain_weights = current_remain_weights - weights
                    next_remain_weights = self._approximate_zero_weights(next_remain_weights)

                    if debug:
                        print(f'[Curr={curr_node},action={action}->Next={next_node}], {current_remain_weights}-{weights}->{next_remain_weights}')

                    a = self._weights_set[next_node].array
                    # Check if next_node has a strategy with the weight size of next_remain_weight
                    if self._in_pareto_front(next_remain_weights,
                                             self._weights_set[next_node].array):
                        strategies[str(pareto_point)][curr_node].add(action)
                        search_queue.put((next_node, next_remain_weights))

        return strategies

    def _in_pareto_front(self, weights, minimal_elements):
        for minimal_element in minimal_elements:
            # if all(weights >= minimal_element):
            if np.allclose(weights, minimal_element, self._atol, self._rtol, self._equal_nan):
                return True
        return False

    def _converged(self, curr: Dict, prev: Dict) -> bool:
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

    def _approximate_zero_weights(self, weights: np.ndarray) -> np.ndarray:
        # VERY HACKY: Converting almost-zero values to string mess things up,
        # So we simply replace them with zero values
        n_weight = len(weights)

        almost_zero_indices = np.isclose(weights, np.zeros(n_weight),
                                         self._atol, self._rtol, self._equal_nan)
        if any(almost_zero_indices):
            weights[almost_zero_indices] = 0.0

        return weights

    def _plot_pareto(self, pareto_points: List, weight_names: List[str] = None,
                     margin_ratio: float = 0.2) -> None:
        n_points = len(pareto_points)

        if n_points == 0:
            warnings.warn('No Pareto Points Found')
            return

        pareto_points = np.array(pareto_points)
        columnIndex = 0
        # Sort 2D numpy array by 2nd Column
        pareto_points = pareto_points[pareto_points[:,columnIndex].argsort()]

        n_dim = len(pareto_points[0])

        if n_dim not in [2, 3]:
            print('We can only visualize 2D or 3D pareto points')

        if len(weight_names) != n_dim:
            raise ValueError(f'No. of weight names must be of a size {n_dim}')

        fig = plt.figure()

        if n_dim == 2:
            ax = fig.add_subplot(111)

            ax.plot(pareto_points[:, 0], pareto_points[:, 1], '-x')
            for p in pareto_points:
                ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', p)
            # limits = plt.axis()

            # # VERY HACKY WAY to add 1 extra node for coloring all solutions until far end
            # max_x_margin = pareto_points[-1, 0] + margin_ratio * (limits[1] - limits[0])
            # max_y_margin = pareto_points[0, 1] + margin_ratio * (limits[3] - limits[2])
            # min_y_pareto = pareto_points[-1, 1]
            # # Add far most right node to a pareto set
            # pareto_points = np.append(pareto_points, [[max_x_margin, min_y_pareto]], axis=0)
            # # Now fill area across X axis between ymin(pareto) and ymax
            # ax.fill_between(pareto_points[:, 0], pareto_points[:, 1], max_y_margin)
            # # Set axis limits to only include the original pareto points
            # plt.axis(limits)

            ax.set_xlabel(weight_names[0])
            ax.set_ylabel(weight_names[1])

        if n_dim == 3:
            ax = fig.add_subplot(111, projection='3d')

            ax.projection(pareto_points[:, 0], pareto_points[:, 1], pareto_points[: 2])

            ax.set_xlabel(weight_names[0])
            ax.set_ylabel(weight_names[1])
            ax.set_zlabel(weight_names[2])

        plt.show()
