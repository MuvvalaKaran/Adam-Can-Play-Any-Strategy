import copy
import queue
import time
import warnings
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Union
from shapely.geometry import Polygon

from ..graph import TwoPlayerGraph
from .value_iteration import ValueIteration
from .adversarial_game import ReachabilityGame
from ..prism import MealyMachine, StochasticStrategy, DeterministicStrategy

class Strategy:
    pass


INFINITY = 1e10
EdgeWeight = List[float]
Point = List[float]
ParetoPoint = List[float]
ParetoPoints = List[ParetoPoint]
Strategies = Dict[str, Strategy]


def get_vertices(polygon: Polygon) -> List:
    # X, Y = polygon.exterior.coords.xy
    X, Y = polygon.exterior.xy

    vertices = set()
    for x, y in zip(X, Y):
        vertices.add((x, y))
    return list(vertices)


class ParetoFront(metaclass=ABCMeta):

    """An array to store elements"""
    _pareto_points = None

    """Whether a first element is added to the array or not"""
    _first_elem_added = False

    """Polyhedron"""
    _polyhedron = None

    """Convex Hull"""
    _convex: bool = False

    """Upper Bound of the convex hull"""
    _upperbound: float = None

    """Whether to keep a polyhedron"""
    _use_polyhedron: bool = False

    """Name of the node"""
    _node_name: str = None

    """Successor Node's front"""
    _successor_fronts: List = None

    def __init__(self,
                 points: Union[int, float, List, np.ndarray] = None,
                 use_polyhedron: bool = False,
                 convex: bool = False,
                 upperbound: float = INFINITY,
                 node_name: str = None):
        """
        Pareto Front Class at each node.
        It keeps either a list of pareto points or a polyhedron.
        The polyhedron could be convex or non-convex.

        :args points:           1D or 2D array
        :args use_polyhedron:   Whether to use a polyhedron
        :args convex:           Whether the polygon would be convex or not
        :args upperbound:       The upperbound of the polyhedron

        e.g.,
            array = [[1], [2]]
            array = [[1,2], [3,4]]
            array = [[1,2,3], [4,5,6]]

        If 1D array is given, it will be converted as 2D
        e.g.,
            array = [1, 2] -> [[1], [2]]
        """
        self._upperbound = INFINITY if upperbound == 0 else upperbound
        self._convex = convex
        self._use_polyhedron = use_polyhedron
        self._node_name = node_name

        if self._use_polyhedron:
            self._polyhedron = self.construct_polyhedron(points, self._upperbound, self._convex)
            points = self._get_polyhedron_vertices()

        self.set_pareto_points(points)

    def construct_polyhedron(self,
        points: Union[List, np.ndarray],
        upperbound: float,
        convex: bool):

        if  points is None or not isinstance(points, (List, np.ndarray)) or len(points) == 0:
            return None

        points = self._check_valid_points(points)

        if convex:
            return self.construct_convex_polyhedron(points, upperbound)
        else:
            return self.construct_nonconvex_polyhedron(points, upperbound)

    def construct_nonconvex_polyhedron(self, points: Union[List, np.ndarray], upperbound: float):
        """
        Construct a non-convex polygon from points of the polygon
        TODO: Currently, limited to polygons (2D Polyhedron) and
        dimensions other than 2D are not supported.
        """
        polygons = []
        for p in points:
            polygon = Polygon([p, (p[0], upperbound), (upperbound, upperbound), (upperbound, p[1])])
            polygons.append(polygon)

        if len(polygons) == 1:
            return polygons[0]

        polygon_union = polygons[0]
        for polygon in polygons[1:]:
            polygon_union = polygon_union.union(polygon)

        return polygon_union

    def construct_convex_polyhedron(self, points: Union[List, np.ndarray], upperbound: float):
        """
        Construct a convex polygon from points of the polygon
        TODO: Currently, limited to polygons (2D Polyhedron) and
        dimensions other than 2D are not supported.
        """
        # Sort by the first column
        columnIndex = 0
        points = np.array(points)
        points = points[points[:,columnIndex].argsort()].tolist()

        points = [[points[0][0], upperbound]] + points + [[upperbound, points[-1][1]], [upperbound, upperbound]]

        return Polygon(points)

    def _get_polyhedron_vertices(self, polyhedron = None, exclude_threshold: bool = True, epsilon: int = 5):
        if polyhedron is None:
            if not self.has_polyhedron():
                return None
            polyhedron = self._polyhedron

        vertices =  get_vertices(polyhedron)

        vertices_wo_threshold = []
        for vertex in vertices:
            vertex = list(vertex)
            if self._upperbound not in vertex:
            # if any(np.around(vertex, epsilon) == self._upperbound):
                vertices_wo_threshold.append(vertex)

        return self._get_minimum_element_set(vertices_wo_threshold)

    def set_pareto_points(self, points: Union[List, np.ndarray]):
        points = self._check_valid_points(points)

        if points is None:
            return

        self._pareto_points = self._get_minimum_element_set(points)

        if not self._first_elem_added:
            self._first_elem_added = True

    def _check_valid_points(self, points: Union[List, np.ndarray]):
        # 1. Check if valid array
        if points is None or len(points) == 0:
            return None

        if isinstance(points, (int, float)):
            points = [[points]]

        if not isinstance(points, (List, np.ndarray)):
            raise Exception('points must be a list or np.ndarray')

        # 2. Check for the dimensions of the array
        def dim(_array):
            """"return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return [len(_array)]+dim(_array[0]) if(isinstance(_array, (List, np.ndarray))) else []
        ndim = len(dim(points))

        # Only accept 1D or 2D
        if ndim >= 3:
            raise ValueError('points be 1D or 2D array')

        # If 1D, make it 2D
        if ndim == 1:
            if isinstance(points, List):
                points = [points]
            else:
                points = np.array([points])

        elem_size = len(points[0])

        # Check if the size of the point is same as the other points
        for point in points:
            if point is None:
                return None

            if not isinstance(point, (List, np.ndarray)):
                raise Exception('A point must be a list or np.ndarray')

            if len(point) != elem_size:
                msg = f'A point of a set must be a size of {elem_size}'
                raise ValueError(msg)

        return points

    def _initialize_pareto_points(self, point: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
        """
        :arg point:   A 1D array point
        """
        elem_size = len(point)
        return np.array([[np.inf] * elem_size])

    def _get_minimum_element_set(self, points: Union[List, np.ndarray]):
        points = np.array(points)
        elem_size = len(points[0])

        delete_indices = []
        for i, current in enumerate(points):
            points_wo_current = copy.deepcopy(points)
            points_wo_current[i] = [-1]*elem_size

            for i, vertex in enumerate(points_wo_current):
                if all(current <= vertex):
                    # delete the element
                    delete_indices.append(i)

        new_points = np.delete(points, delete_indices, axis=0)

        return new_points

    @abstractmethod
    def update(self, successor_fronts: List,
               edge_weights: List[EdgeWeight]):
        raise NotImplementedError('"update" function is not implemented')

    def expand_by(self, distance: List[float]) -> None:
        if not self.use_polyhedron():
            raise Exception('use_polyhedron is set to false')

        if not self.has_polyhedron():
            return
        # TODO: Expand the current polyhedron by given distance

        # 1. get the pareto points.
        pareto_points = self._get_polyhedron_vertices()

        # Get the dimension of the polyhedron
        ndim = len(pareto_points[0])

        # Check if the given distance matches with the dimension of the polyhedron
        if len(distance) != ndim:
            raise Exception(f'The dimension of {distance} must be {ndim}')

        # 2. Expanded the pareto points by distance
        pareto_points = [(np.array(pareto_point) + np.array(distance)).tolist() for pareto_point in pareto_points]

        # 3. Construct the polyhedron from the expanded points
        self._polyhedron = self.construct_polyhedron(pareto_points, self._upperbound, self._convex)

    def _expand_successor_pareto_points(self,
            successor_fronts: List,
            edge_weights: List[EdgeWeight]) -> List[List[Tuple]]:
        """
        Compute for a list of successor's pareto points
        """

        successor_pareto_points = []
        for front, weight in zip(successor_fronts, edge_weights):
            if not front.is_initial_pareto_updated():
                continue

            pareto_points = []
            for pareto_point in front.pareto_points:
                curr_node_weight = pareto_point + weight
                pareto_points.append(curr_node_weight.tolist())

            successor_pareto_points.append(pareto_points)

        return successor_pareto_points

    def _expand_successor_pareto_fronts(self, successor_fronts: List,
            edge_weights: List[EdgeWeight]) -> List[List[Tuple]]:
        """
        Compute for a list of successor's pareto points
        """
        successor_pareto_fronts = []
        for front, weight in zip(successor_fronts, edge_weights):
            if not front.is_initial_pareto_updated():
                continue

            front_ = copy.deepcopy(front)
            front_.expand_by(weight)
            successor_pareto_fronts.append(front_)

        # self._successor_fronts = successor_pareto_fronts

        return successor_pareto_fronts

    def __str__(self):
        return str(self.pareto_points)

    def use_polyhedron(self):
        return self._use_polyhedron

    def has_polyhedron(self):
        return self._polyhedron is not None

    def has_successor_polyhedra(self):
        if self._successor_fronts is None:
            return False
        return True

    def is_initialized(self) -> bool:
        return self._pareto_points is not None

    def is_initial_pareto_updated(self) -> bool:
        """
        Checks whether the initial values were overwritten by incoming elements
        """
        return self._first_elem_added

    def __eq__(self, others):
        return np.array_equal(self.pareto_points, others.pareto_points)

    def plot(self, include_successors: bool = True):
        # Check if the polyhedron is initialized
        if not self.has_polyhedron():
            return

        # Check if the successor polyhedron is saved
        if include_successors and not self.has_successor_polyhedra():
            return

        plot = self._plot_2d_polyhedron

        if include_successors:
            n_front = len(self._successor_fronts) + 1 # +1 for plotting its own polyhedron

            vertices = self._get_polyhedron_vertices()
            axislims = np.max(vertices, axis=0) + np.ones(len(vertices[0]))

            fig = plt.figure(figsize=(4 * n_front, 4))
            last_ax = fig.add_subplot(1, n_front, n_front)
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # Plot sucessors
            for i, front in enumerate(self._successor_fronts):
                ax = fig.add_subplot(1, n_front, i+1)
                plot(front.polyhedron, axislims,
                    ax=ax, color=colors[i], title=front.node_name)
                plot(front.polyhedron, axislims,
                    ax=last_ax, color=colors[i], title=front.node_name, ls='--')

            # Plot its own
            plot(self.polyhedron, axislims,
                ax=last_ax, color=colors[n_front-1], title=self.node_name)
            return

        plot(self.polyhedron, title=self.node_name)

    def _plot_2d_polyhedron(self,
            polyhedron, axislims: List[float]=None,
            ax=None, ls='-', color='C0', title=None):
        if polyhedron is None:
            return

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.axvline(x=0, ymin=0, color='k')
        ax.axhline(y=0, xmin=0 ,color='k')

        X, Y = polyhedron.exterior.xy
        ax.plot(X, Y, color=color, ls=ls)
        ax.set_aspect('equal')

        # Set axis limits
        vertices = self._get_polyhedron_vertices(polyhedron)
        if axislims is None or len(axislims)!=len(vertices[0]) :
            axislims = np.max(vertices, axis=0) + np.ones(len(vertices[0]))

        ax.set_xlim([-1, axislims[0]])
        ax.set_ylim([-1, axislims[1]])

        if title is not None:
            ax.set_title(title)

    @property
    def pareto_points(self) -> List[List[float]]:
        if not self.is_initialized():
            return []
        return self._pareto_points

    @property
    def polyhedron(self):
        if not self.has_polyhedron():
            raise Exception('Please initialize the polyhedron first')
        return self._polyhedron

    @property
    def node_name(self) -> str:
        if self._node_name is None:
            return ''
        return self._node_name


class SysParetoFront(ParetoFront):

    def __init__(self,
                 points: Union[int, float, List, np.ndarray] = None,
                 use_polyhedron: bool = True,
                 stochastic: bool = False,
                 upperbound: float = INFINITY,
                 node_name: str = None):
        super().__init__(points,
                         use_polyhedron,
                         convex=stochastic,
                         upperbound=upperbound,
                         node_name=node_name)

    def update(self, successor_fronts: List[ParetoFront],
               edge_weights: List[EdgeWeight]):

        if len(successor_fronts) != len(edge_weights):
            raise Exception('No. of successors should be same as the No. of edges')

        if self.use_polyhedron():
            successor_pareto_fronts = \
                self._expand_successor_pareto_fronts(successor_fronts, edge_weights)

            if len(successor_pareto_fronts) == 0:
                return

            # Take an union of the successor pareto fronts
            self._polyhedron = self.union_of_successor_fronts(successor_pareto_fronts)
            pareto_points = self._get_polyhedron_vertices()
        else:
            successor_pareto_points = \
                self._expand_successor_pareto_points(successor_fronts, edge_weights)

            pareto_points = self.union_of_successor_pareto_points(
                successor_pareto_points)

        self.set_pareto_points(pareto_points)

    def union_of_successor_pareto_points(self, successor_pareto_points: List[List[Tuple]]):
        """From a list of successor's pareto points, compute the pareto points for the predecessor"""
        # Upon initialization, compute for the upper bound
        if len(successor_pareto_points) == 0:
            return []

        # Merge lists
        new_pareto_points = []
        for pareto_points in successor_pareto_points:
            new_pareto_points += pareto_points

        return new_pareto_points

    def union_of_successor_fronts(self, successor_pareto_fronts: List):
        if len(successor_pareto_fronts) == 0:
            return None

        if len(successor_pareto_fronts) == 1:
            return successor_pareto_fronts[0].polyhedron

        # Take intersections of all those polyhedra
        polyhedron = successor_pareto_fronts[0].polyhedron
        for front in successor_pareto_fronts[1:]:
            polyhedron = polyhedron.union(front.polyhedron)

        return polyhedron


class EnvParetoFront(ParetoFront):

    def __init__(self,
                 points: Union[int, float, List, np.ndarray] = None,
                 use_polyhedron: bool = True,
                 stochastic: bool = False,
                 upperbound: float = INFINITY,
                 node_name: str = None):
        super().__init__(points,
                         use_polyhedron,
                         convex=stochastic,
                         upperbound=upperbound,
                         node_name=node_name)

    def update(self, successor_fronts: List[ParetoFront],
               edge_weights: List[EdgeWeight]):
        """
        In this function, our goal is to find the minimum maximum pareto points
        that the system player can guarantee to achieve.
        In other word, while the environment player tried to play adversarial
        (maximum weights), the system player needs to find a value set that
        can be reached at any successor node.
        """
        if len(successor_fronts) != len(edge_weights):
            raise Exception('No. of successors should be same as the No. of edges')

        if self._use_polyhedron:
            successor_pareto_fronts = \
                self._expand_successor_pareto_fronts(successor_fronts, edge_weights)

            if len(successor_pareto_fronts) == 0:
                return

            # Take an intersection of the successor pareto fronts
            self._polyhedron = self.intersection_of_successor_fronts(successor_pareto_fronts)
            pareto_points = self._get_polyhedron_vertices()
        else:
            successor_pareto_points = \
                self._expand_successor_pareto_points(successor_fronts, edge_weights)

            pareto_points = self.intersection_of_successor_pareto_points(
                successor_pareto_points)

        self.set_pareto_points(pareto_points)

    def intersection_of_successor_pareto_points(self, successor_pareto_points: List[List[Tuple]]):
        """
        From a list of successor's pareto points, compute the pareto points for the predecessor
        """
        # Upon initialization, compute for the upper bound
        if len(successor_pareto_points) == 0:
            return []

        # First construct a list of polyhedra at each node
        # Each successor should be keeping its own polyhedron, so you could just access those polyhedra
        polyhedrons = []
        for pareto_points in successor_pareto_points:
            # Either construct it from the pareto points
            polyhedron = self.construct_polyhedron(pareto_points, self._upperbound, self._convex)
            # Or Get the pareto points
            # polyhedron = successor_node.get_polygon()
            polyhedrons.append(polyhedron)

        # Take intersections of all those polygons
        P = polyhedrons[0]
        for polyhedron in polyhedrons[1:]:
            P = P.intersection(polyhedron)

        # get the predecessor's pareto points
        intersections = self._get_polyhedron_vertices(P)

        return intersections

    def intersection_of_successor_fronts(self, successor_pareto_fronts: List):
        if len(successor_pareto_fronts) == 0:
            return []

        if len(successor_pareto_fronts) == 1:
            return successor_pareto_fronts[0].polyhedron

        # Take intersections of all those polyhedrons
        polyhedron = successor_pareto_fronts[0].polyhedron
        for front in successor_pareto_fronts[1:]:
            polyhedron = polyhedron.intersection(front.polyhedron)

        return polyhedron


class MultiObjectiveSolver:
    """
    Currently, only supported for two weights, weight and auto_weight.
    TODO: Support for multiple objective functions

    :attribute _Wg_graph: Graph with Game Weights
    :attribute _Wa_graph: Graph with Automaton Weights
    """
    def __init__(self, graph: TwoPlayerGraph, rtol=1e-05, atol=1e-08, equal_nan=False,
                 remove_cycles: bool = True):
        """
        """
        self._game = copy.deepcopy(graph)
        self._rtol = rtol
        self._atol = atol
        self._equal_nan = equal_nan

        self._reachability_solver = ReachabilityGame(game=self._game)
        self._reachability_solver.reachability_solver()
        self._game.delete_selfloops()

        self._pareto_fronts = None
        self._strategies = None
        self._upperbounds = self._compute_upperbound_for_pareto_points()

    def _compute_upperbound_for_pareto_points(self) -> List[float]:
        """
        Compute the maximum possible values that can be achieved in the game
        """
        max_depth = len(self._game._graph.nodes())

        max_weights = []
        for name in self._game.weight_types:
            weights = [e[2]['weights'].get(name) for e in self._game._graph.edges.data()]
            max_single_weight = max(weights)
            max_weights.append(max_depth * max_single_weight)

        return max_weights

    def get_strategies(self):
        if self._strategies is None:
            raise Exception("No strategies found. Please run 'solve' or 'solve' first.")

        return self._strategies

    def get_a_strategy_for(self, pareto_point: Union[str, ParetoPoint]):
        strategies = self.get_strategies()

        if not isinstance(pareto_point, str):
            pareto_point = str(pareto_point)

        if pareto_point not in strategies:
            raise Exception(f'{pareto_point} is not in the comptued pareto points')

        return strategies[pareto_point]

    def get_pareto_points(self):
        if self._pareto_fronts is None:
            raise Exception("No pareto points found. Please run 'solve' function first.")

        init_node = self._game.get_initial_states()[0][0]
        return self._pareto_fronts[init_node].pareto_points

    def get_pareto_point_at(self, node):
        if self._pareto_fronts is None:
            raise Exception("No pareto points found. Please run 'solve' function first.")

        if node not in self._game._graph.nodes():
            raise Exception(f'{node} is not in the game graph')

        return self._pareto_fronts[node].pareto_points

    def is_pareto_computed(self) -> bool:
        return self._pareto_fronts is not None

    def check_if_pareto_computed(self) -> None:
        if not self.is_pareto_computed():
            msg = "No pareto fronts found. Please run 'solve' or 'solve_pareto_points' first."
            raise Exception(msg)

    def solve(self,
              stochastic: bool=False,
              adversarial: bool=False,
              bound: Point = None,
              all_strategies: bool = True,
              plot_pareto: bool = False,
              plot_all_paretos: bool = False,
              plot_strategies: bool = False,
              plot_graph_with_pareto: bool = True,
              decimals: int = 2,
              debug: bool = False) -> Tuple[ParetoPoints, Strategies]:

        pareto_points = self.solve_pareto_points(
            stochastic, adversarial, plot_pareto, plot_all_paretos,
            plot_graph_with_pareto, decimals, debug)

        strategies = self.solve_strategies(
            bound, all_strategies, plot_strategies, debug)

        return pareto_points, strategies

    def solve_pareto_points(self, stochastic: bool, adversarial: bool = True,
        plot_pareto: bool = True, plot_all_paretos: bool = False,
        plot_graph_with_pareto: bool = True, decimals: int = 2,
        debug: bool = False) -> ParetoPoints:

        self._stochastic = stochastic

        # Pareto Computation
        pareto_points = self._compute_pareto_points(stochastic, adversarial, debug)
        self._label_pareto_on_graph()

        # Pareto Visualization
        if plot_pareto:
            self._plot_pareto_front()

        if plot_all_paretos:
            self._plot_pareto_fronts()

        if plot_graph_with_pareto:
            self._game.plot_graph()

        return pareto_points

    def solve_strategies(self, bound: Point = None, all_strategies: bool = False,
        plot_strategies: bool = True, debug: bool = False) -> Strategies:

        self.check_if_pareto_computed()

        if bound is None and not all_strategies:
            msg = 'Please either provide a constraint to compute a strategy for a specific point in the pareto front or choose "all_strategies" to compute strategies for all pareto points'
            raise ValueError(msg)

        pareto_points = self.get_pareto_points()

        if bound is not None:
            pareto_points = self._compute_points_in_pareto_front(bound, pareto_points)

        strategies = {}
        for pareto_point in pareto_points:
            strategy = self._compute_strategy(pareto_point, self._stochastic, debug)
            strategies[tuple(pareto_point)] = strategy

        self._strategies = strategies

        return strategies

    def _compute_points_in_pareto_front(self,
        bound: Point,
        pareto_points: ParetoPoints) -> List[Point]:

        """
        Given a constraint and a pareto front at the initial state,
        compute a point in the pareto front
        """

        points = []
        for pareto_point in pareto_points:
            if all(np.array(bound) >= np.array(pareto_point)):
                points.append(pareto_point.tolist())

        if len(points) != 0:
            return points

        init_node = self._game.get_initial_states()[0][0]
        pareto_front = copy.deepcopy(self._pareto_fronts[init_node])

        p = bound
        polygon = Polygon([(0, 0), (p[0], 0), p, (0, p[1])])

        # Intersection
        polygon = polygon.intersection(pareto_front._polyhedron)
        points = pareto_front._get_polyhedron_vertices(polygon)

        return points

    # Pareto Points Computation
    def _compute_pareto_points(self,
        stochastic: bool, adversarial: bool = True, debug: bool = True):

        #################### Winning Region Computation ####################

        accept_node = self._game.get_accepting_states()[0]

        #################### Pareto Points Computation ####################

        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        n_weight = len(weights)

        pareto_fronts = {}

        for node in self._game._graph.nodes():

            player = self._game.get_state_w_attribute(node, "player")

            if player == 'eve': # System / Robot
                pareto_fronts[node] = SysParetoFront(
                    stochastic=stochastic, node_name=node)
            else:
                if adversarial:
                    pareto_fronts[node] = EnvParetoFront(
                        upperbound=max(self._upperbounds), stochastic=stochastic, node_name=node)
                else:
                    pareto_fronts[node] = SysParetoFront(
                        upperbound=max(self._upperbounds), stochastic=stochastic, node_name=node)

        pareto_fronts[accept_node] = SysParetoFront(
            np.zeros(n_weight), upperbound=max(self._upperbounds), stochastic=stochastic, node_name=accept_node)

        pareto_fronts_prev = None

        # TODO: Identify Cycles with 0 weights
        cycles = self._game.get_cycles()
        node_to_cycles = defaultdict(lambda: [])
        for i, cycle in enumerate(cycles):
            for node in cycle:
                node_to_cycles[node].append(i)

        visitation_order = self._decide_visitation_order()
        strategies_at_node = defaultdict(lambda: defaultdict(lambda: set()))

        iter_var = 0
        while not self._converged(pareto_fronts, pareto_fronts_prev):

            if debug:
                print(f"{iter_var} Iterations on a graph with {len(visitation_order)} nodes")

            pareto_fronts_prev = copy.deepcopy(pareto_fronts)
            iter_var += 1

            for u_node in visitation_order:

                # Only allow transitions to the winning region
                if u_node not in self._reachability_solver.sys_winning_region:
                    continue

                player = self._game.get_state_w_attribute(u_node, "player")
                fronts = []
                edge_weights = []

                for v_node in self._game._graph.successors(u_node):

                    # Avoid self loop
                    if u_node == v_node:
                        continue

                    # get u to v's edge weight
                    edge_weight = self._game.get_edge_attributes(u_node, v_node, 'weights')
                    edge_weight = list(edge_weight.values())

                    fronts.append(pareto_fronts[v_node])
                    edge_weights.append(edge_weight)

                pareto_fronts[u_node].update(fronts, edge_weights)

        self._pareto_fronts = pareto_fronts
        init_node = self._game.get_initial_states()[0][0]
        pareto_points = self._pareto_fronts[init_node].pareto_points

        return pareto_points

    def _decide_visitation_order(self):
        accept_node = self._game.get_accepting_states()[0]

        search_queue = queue.Queue()
        visited = set()
        visitation_order = []

        search_queue.put(accept_node)
        visited.add(accept_node)
        visitation_order.append(accept_node)

        while not search_queue.empty():

            v_node = search_queue.get()

            for u_node in self._game._graph.predecessors(v_node):
                # Avoid self loop
                if u_node == v_node:
                    continue

                if u_node not in visited:
                    visitation_order.append(u_node)
                    visited.add(u_node)
                    search_queue.put(u_node)

        return visitation_order

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

    def _label_pareto_on_graph(self, decimals: int = 2) -> None:

        self.check_if_pareto_computed()

        node_labels = {}
        for node in self._game._graph.nodes():
            pareto_points = self._pareto_fronts[node].pareto_points

            node_labels[node] = np.array2string(np.array(pareto_points).round(decimals))
        self._game.set_node_labels_on_fancy_graph(node_labels)

    def _plot_pareto_front(self) -> None:

        self.check_if_pareto_computed()

        accept_node = self._game.get_accepting_states()[0]
        weights = self._game.get_edge_attributes(accept_node, accept_node, 'weights')
        weight_names = list(weights.keys())

        init_node = self._game.get_initial_states()[0][0]

        self._pareto_fronts[init_node].plot(include_successors=False)

        plt.xlabel(weight_names[0])
        plt.ylabel(weight_names[1])

        plt.show()

    def _plot_pareto_fronts(self, include_successors: bool = True) -> None:

        self.check_if_pareto_computed()

        for node in self._game._graph.nodes():
            self._pareto_fronts[node].plot(include_successors)
            plt.show()

    # Strategy Computation
    def _compute_strategy(self, pareto_point: ParetoPoint, stochastic: bool,
        debug: bool = False) -> Strategy:
        # Strategy Synthesis
        if stochastic:
            strategy = StochasticStrategy.from_pareto_fronts(
                self._game,
                self._pareto_fronts,
                init_point=pareto_point)
        else:
            strategy = DeterministicStrategy.from_pareto_fronts(
                self._game,
                self._pareto_fronts,
                init_pareto_point=pareto_point)

        strategy.plot_graph()
        strategy.plot_original_graph()

        return strategy
