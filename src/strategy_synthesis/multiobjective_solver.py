import copy
import _pickle as cPickle
import queue
import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Union, Optional
import ppl
from ppl import Variable, Generator_System, C_Polyhedron, point, Constraint_System

from ..graph import TwoPlayerGraph
from .value_iteration import ValueIteration
from .adversarial_game import ReachabilityGame


def _mpzs_to_list(mpzs):
    """
    Convert a list of mpzs to a list of floats
    e.g.) [mpz(4), mpz(1)] to [4. 1.]
    """
    return list(map(float, mpzs))


def get_point(generator: ppl.Generator):
    return _mpzs_to_list(generator.coefficients())


def get_points(generators: List[ppl.Generator]):
    return [get_point(g) for g in generators]


def get_minimized_generators(polyhedron: C_Polyhedron):
    return [gen for gen in list(polyhedron.minimized_generators())]


def get_minimized_points(polyhedron: C_Polyhedron):
    generators = get_minimized_generators(polyhedron)
    return get_points(generators)


class ConvexHull(metaclass=ABCMeta):

    """The size of each element in an array"""
    _n_dim = None

    """_n_dim of PPL Variables"""
    _variables = None

    """The polyhedra"""
    _polyhedra = None

    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        """
        :args array:        1D or 2D array
        e.g.,
            array = [[1], [2]]
            array = [[1,2], [3,4]]
            array = [[1,2,3], [4,5,6]]

        If 1D array is given, it will be stored as 2D
        e.g.,
            array = [1, 2] -> [[1], [2]]
        """
        # Check the dimensions of the given array and turn it into a 2D array
        array = self._check_valid_array(array)

        # The array should always be either 2D array or None
        if array is not None:
            for arr in array:
                self.add(arr)

    def _check_valid_array(self, array: Union[int, float, List, np.ndarray]) \
                          -> Union[List, np.ndarray]:
        """
        Check for the dimensions of an array and turn it into a 2D array.
        It will raise an error if initialization fails.

        :arg array:         Ideally a 2D array.
                            It could be a singular value (int or float)
                            or a 1D/2D array (list or np.ndarray)
        :return:            A 2D array if initialized correctly, None otherwise
        """
        # 1. Check if valid array
        if array is None:
            return None

        if isinstance(array, (int, float)):
            array = [[array]]

        if not isinstance(array, (List, np.ndarray)):
            raise TypeError('An array has to be a list or np.array')

        # 2. Check for the dimensions of the array
        def dim(_array):
            """"return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return [len(_array)]+dim(_array[0]) if(isinstance(_array, (List, np.ndarray))) else []
        ndim = len(dim(array))

        # Only accept 1D or 2D
        if ndim >= 3:
            raise ValueError('An array has to be 1D or 2D array')

        # If 1D, make it 2D
        if ndim == 1:
            if isinstance(array, List):
                array = [array]
            else:
                array = np.array([array])

        return array

    def _initialize(self, element: Union[List, np.ndarray]) -> None:
            self._n_dim = len(element)
            self._variables = [Variable(i) for i in range(self._n_dim)]
            self._polyhedra = self._create_polyhedron([0]*self._n_dim)

    def add(self, element: Union[int, float, List, np.ndarray]):
        """
        Add an element (point) to the convex hull
        """
        element = self._sanity_check(element)
        self._add_point(element)

    def take_union(self, element: Union[int, float, List, np.ndarray]) -> None:
        """
        Add an element (point) to the convex hull
        """
        element = self._sanity_check(element)
        self._take_union(element)

    def take_intersection(self, element: Union[int, float, List, np.ndarray]) -> None:
        """
        Add an element (point) to the convex hull
        """
        element = self._sanity_check(element)
        self._take_intersection(element)

    @abstractmethod
    def _add_point(self, element: Union[int, float, List, np.ndarray]) -> None:
        raise NotImplementedError('"_add_point" function is not implemented')

    def _create_polyhedron(self, element: Union[List, np.ndarray]):
        cs = Constraint_System()

        for i in range(self._n_dim):
            cs.insert(self._variables[i] >= element[i])

        return C_Polyhedron(cs)

    def _take_union(self, element: Union[int, float, List, np.ndarray]) -> None:
        """
        Add a point to the current convex hull
        """
        # Take the union with the existing convex hull
        polyhedron = self._create_polyhedron(element)
        self._polyhedra.poly_hull_assign(polyhedron)

    def _take_intersection(self, element: Union[int, float, List, np.ndarray]) -> None:
        """
        Add a point to the current convex hull
        """
        # Take the intersection with the existing convex hull
        polyhedron = self._create_polyhedron(element)
        self._polyhedra.intersection_assign(polyhedron)

    def _sanity_check(self, element: Union[int, float, List, np.ndarray]) \
                      -> Union[List, np.ndarray]:
        # Check if element is not None
        if element is None:
            raise ValueError('An element cannot be None')

        # If a single value is given, make it a 1D array
        if isinstance(element, (int, float)):
            element = [element]

        # The element type must be of a list or np.ndarray
        if not isinstance(element, (List, np.ndarray)):
            raise TypeError('An element has to be a list or np.array')

        # 2. Check for the dimensions of the array
        def dim(_array):
            """"return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return [len(_array)]+dim(_array[0]) if(isinstance(_array, (List, np.ndarray))) else []
        ndim = len(dim(element))

        # Only accept a 1D array
        if ndim != 1:
            raise ValueError('An element must be a 1D array')

        # Initialized the element size, if not yet initialized
        if not self.initialized:
            self._initialize(element)

        # Check if the size of the element is same as the other elements
        if len(element) != self._n_dim:
            msg = f'An element of a set must be a size of {self._n_dim}'
            raise ValueError(msg)

        return element

    @property
    def initialized(self) -> bool:
        return self._n_dim is not None

    @property
    def array(self):
        # if not self.initialized:
        #     return []
        return np.array(get_minimized_points(self._polyhedra))

    @property
    def n_elem(self):
        if not self.initialized:
            return 0
        return len(self.array)

    # TODO: This does not suit well with convex hull. Better implement it for ElementSet.
    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.n_elem:
            result = self.array[self._i]
            self._i += 1
            return result
        else:
            raise StopIteration

    def __eq__(self, others):
        return np.array_equal(self.array, others.array)

    def __str__(self):
        return str(self.array)


class UnionConvexHull(ConvexHull):

    # def __init__(self, **kwargs):
        # super().__init__(**kwargs)
    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        super().__init__(array)


    def _add_point(self, point: List) -> None:
        """
        Add a point to the current convex hull
        """
        self._take_union(point)


class IntersectionConvexHull(ConvexHull):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        super().__init__(array)

    def _add_point(self, point: List) -> None:
        """
        Add a point to the current convex hull
        """
        self._take_intersection(point)


class ElementSet(metaclass=ABCMeta):

    """An array to store elements"""
    _array = None

    """The size of each element in an array"""
    _elem_size = None

    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        """
        :args array:        1D or 2D array
        e.g.,
            array = [[1], [2]]
            array = [[1,2], [3,4]]
            array = [[1,2,3], [4,5,6]]

        If 1D array is given, it will be stored as 2D
        e.g.,
            array = [1, 2] -> [[1], [2]]
        """
        # Check the dimensions of the given array and turn it into a 2D array
        array = self._check_valid_array(array)

        # The array should always be either 2D array or None
        if array is not None:
            for arr in array:
                self.add(arr)

    def _check_valid_array(self, array: Union[int, float, List, np.ndarray]) \
                          -> Union[List, np.ndarray]:
        """
        Check for the dimensions of an array and turn it into a 2D array.
        It will raise an error if initialization fails.

        :arg array:         Ideally a 2D array.
                            It could be a singular value (int or float)
                            or a 1D/2D array (list or np.ndarray)
        :return:            A 2D array if initialized correctly, None otherwise
        """
        # 1. Check if valid array
        if array is None:
            return None

        if isinstance(array, (int, float)):
            array = [[array]]

        if not isinstance(array, (List, np.ndarray)):
            raise TypeError('An array has to be a list or np.array')

        # 2. Check for the dimensions of the array
        def dim(_array):
            """"return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return [len(_array)]+dim(_array[0]) if(isinstance(_array, (List, np.ndarray))) else []
        ndim = len(dim(array))

        # Only accept 1D or 2D
        if ndim >= 3:
            raise ValueError('An array has to be 1D or 2D array')

        # If 1D, make it 2D
        if ndim == 1:
            if isinstance(array, List):
                array = [array]
            else:
                array = np.array([array])

        return array

    def _sanity_check(self, element: Union[int, float, List, np.ndarray]) \
                      -> Union[List, np.ndarray]:
        # Check if element is not None
        if element is None:
            raise ValueError('An element cannot be None')

        # If a single value is given, make it a 1D array
        if isinstance(element, (int, float)):
            element = [element]

        # The element type must be of a list or np.ndarray
        if not isinstance(element, (List, np.ndarray)):
            raise TypeError('An element has to be a list or np.array')

        # 2. Check for the dimensions of the array
        def dim(_array):
            """"return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return [len(_array)]+dim(_array[0]) if(isinstance(_array, (List, np.ndarray))) else []
        ndim = len(dim(element))

        # Only accept a 1D array
        if ndim != 1:
            raise ValueError('An element must be a 1D array')

        # Initialized the element size, if not yet initialized
        if self._elem_size is None:
            self._elem_size = len(element)

        # Check if the size of the element is same as the other elements
        if len(element) != self._elem_size:
            msg = f'An element of a set must be a size of {self._elem_size}'
            raise ValueError(msg)

        return element

    @abstractmethod
    def add(self, element):
        raise NotImplementedError('"add" function is not implemented')

    def __call__(self):
        return self._array

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.n_elem:
            result = self._array[self._i]
            self._i += 1
            return result
        else:
            raise StopIteration

    def __eq__(self, others):
        return np.array_equal(self._array, others._array)

    def __str__(self):
        return str(self._array)

    @property
    def array(self):
        return self._array

    @property
    def n_elem(self):
        return len(self._array)


class MinimumElementSet(ElementSet):

    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        super().__init__(array)

    def add(self, element: Union[int, float, List, np.ndarray]):
        """
        Add an element to the existing minimum element set (_array)

        Definition of a Minimum Element
            At node s, let two multi-dimensional weights be e and e'.
            We define minimum element as
                (s,e) <= (s', e') iff s=s' and e <= e'

        In other word, weight e has to be less than or equal to e' for all dimensions.
        """
        # Check if the element is a valid array
        element = self._sanity_check(element)

        if self._array is None:
            self._array = np.array([[np.inf]*self._elem_size])

        # If an array already exists, check if it is a minimum element
        if self._check_if_minimum(element):
            # Remove all elements that are larger than the minimum_element
            delete_indices = []
            for i, arr in enumerate(self._array):
                if all(element <= arr):
                    # deletethe element
                    delete_indices.append(i)

            self._array = np.append(self._array, [element], axis=0)
            self._array = np.delete(self._array, delete_indices, axis=0)

    def _check_if_minimum(self, element: np.ndarray):
        # TODO: Without using for loops
        for arr in self._array:
            if all(element >= arr):
                # Mark it as not minimum
                return False

        return True


class MaximumElementSet(ElementSet):

    def __init__(self, array: Union[int, float, List, np.ndarray] = None):
        super().__init__(array)

    def add(self, element: Union[int, float, List, np.ndarray]):
        """
        Add an element to the existing maximum element set (_array)

        Definition of a Maximum Element
            At node s, let two multi-dimensional weights be e and e'.
            We define maximum element as
                (s,e) >= (s', e') iff s=s' and e >= e'

        In other word, weight e has to be greater than or equal to e' for all dimensions.
        """
        # Check if the element is a valid array
        element = self._sanity_check(element)

        if self._array is None:
            self._array = np.array([[0]*self._elem_size])

        # If an array already exists, check if it is a minimum element
        if self._check_if_maximum(element):
            # Remove all elements that are larger than the minimum_element
            delete_indices = []
            for i, arr in enumerate(self._array):
                if all(element >= arr):
                    # deletethe element
                    delete_indices.append(i)

            self._array = np.append(self._array, [element], axis=0)
            self._array = np.delete(self._array, delete_indices, axis=0)

    def _check_if_maximum(self, element: np.ndarray):
        # TODO: Without using for loops
        for arr in self._array:
            if all(element <= arr):
                # Mark it as not minimum
                return False

        return True


class IntersectionConvexHullForDeterministicStrategy(IntersectionConvexHull):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, element: Union[List, np.ndarray]) -> None:
            self._n_dim = len(element)
            self._variables = [Variable(i) for i in range(self._n_dim)]
            self._polyhedra = [self._create_polyhedron([0]*self._n_dim)]

    def _take_intersection(self, element: Union[int, float, List, np.ndarray]) -> None:
        """
        Add a point to the current convex hull
        """
        # Prepare multiple polyhedra (each vertex & new element) -> Get a new vertex
        polyhedra = []
        new_polyhedron = self._create_polyhedron(element)

        for vertex in self.array:
            polyhedron = self._create_polyhedron(vertex)
            polyhedron.intersection_assign(new_polyhedron)
            polyhedra.append(polyhedron)

        self._polyhedra = polyhedra

    @property
    def array(self):
        return np.array([get_minimized_points(p) for p in self._polyhedra])


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
                    debug: bool = False):

        # Pareto Computation
        pareto_points = self._compute_pareto_points(debug)
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

    def _compute_pareto_points(self, debug: bool = True):

        #################### Winning Region Computation ####################

        accept_node = self._game.get_accepting_states()[0]

        #################### Pareto Points Computation ####################

        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        n_weight = len(weights)

        weights_set = {}
        for node in self._reachability_solver.sys_winning_region:
            player = self._game.get_state_w_attribute(node, "player")
            if player == 'eve': # System / Robot
                # weights_set[node] = MinimumElementSet()
                weights_set[node] = UnionConvexHull(np.zeros(n_weight))
            else:
                # weights_set[node] = MaximumElementSet()
                weights_set[node] = IntersectionConvexHull(np.zeros(n_weight))
        # weights_set[accept_node] = MinimumElementSet(np.zeros(n_weight))
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
        element_set = self._weights_set[n_init]
        pareto_points = element_set.array

        return pareto_points

    def _compute_strategies(self, debug: bool = False):
        """
        """
        if self._weights_set is None:
            raise Exception('Pareto Points are not computed')

        n_init = self._game.get_initial_states()[0][0]
        element_set = self._weights_set[n_init]
        pareto_points = element_set.array

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
