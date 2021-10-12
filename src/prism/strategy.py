import copy
import queue
import random
import numpy as np
import networkx as nx
from enum import IntEnum
from bidict import bidict
from graphviz import Digraph
from scipy.optimize import linprog
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from scipy.stats import rv_discrete, uniform
from typing import Union, List, Dict, Tuple, Set, Any, Hashable

from ..graph import Graph

Node = Hashable
Nodes = List[Node]
Action = str
Actions = List[Action]
ParetoPoint = Tuple[float, ...]
# ParetoPoint = str
Point = List[float]
Probability = float
DistOverPareto = Dict[ParetoPoint, Probability]

VIRTUAL_INIT = 'Init'
ACCEPT = 'Accepting'


def solve_memory_trans_probabilities(
    curr_pareto_point: List, edge_weight: List, next_pareto_points: List[List]):
    """
    Compute for the memory transition probabilities by solving the linear programming problem

    :arg curr_pareto_point:     A d-dimension pareto point
    :arg edge_weight:           A d-dimension edge weight
    :arg next_pareto_points:    A list of n pareto points of d-dimension
    """

    # Assume non empty

    # Assume all dimensions are correct

    curr_pareto_point = np.array(curr_pareto_point)
    edge_weight = np.array(edge_weight)
    next_pareto_points = np.array(next_pareto_points)
    c = -np.sum(next_pareto_points, axis=1)

    A_condition = next_pareto_points.T
    b_condition = curr_pareto_point - edge_weight

    n_vertice = next_pareto_points.shape[0]

    A_ub = np.concatenate((np.eye(n_vertice), -np.eye(n_vertice), A_condition))
    b_ub = np.concatenate((np.ones(n_vertice), np.zeros(n_vertice), b_condition))

    A_eq = np.array([np.ones(n_vertice)])   # Sum of probabilities must be 1
    b_eq = np.array([1])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    if not res.success:
        return None

    return res.x


def in_pareto_front(weights, minimal_elements, epsilon: float = 5):
    for minimal_element in minimal_elements:
        if all(np.around(weights, epsilon) >= np.around(minimal_element, epsilon)):
            return True
    return False


class Strategy(Graph):

    def __init__(self, game, graph_name: str):
        super().__init__(config_yaml=None, save_flag=True)
        self._game = copy.deepcopy(game)
        self._graph_name = graph_name

    def _is_graph_constructed(self):
        return self._graph is not None

    def check_if_graph_constructed(self):
        if not self._is_graph_constructed():
            raise Exception('Graph is not yet constructed')

    def plot_graph(self, save_yaml: bool = False, **kwargs):
        """
        A helper method to dump the graph data to a yaml file, read the yaml file and plotting the graph itself
        :return: None
        """
        self.check_if_graph_constructed()

        if save_yaml:
            # dump to yaml file
            self.dump_to_yaml()
            # read the yaml file
            self.read_yaml_file()
        else:
            self._update_graph_yaml()

        # plot it
        self.fancy_graph(**kwargs)

    # @abstractmethod
    def construct_graph(self):
        raise NotImplementedError('Please implement the function')

    # @abstractmethod
    def fancy_graph(self, color=("lightgrey", "red", "purple"), **kwargs) -> None:
        pass

    def get_edges_on_original_graph(self):
        return self._graph.edges()

    # @abstractmethod
    def step(self, state):
        raise NotImplementedError('Please implement the function')

    # @abstractmethod
    def available_actions(self, state):
        raise NotImplementedError('Please implement the function')


class DeterministicStrategy(Strategy):

    def __init__(self,
        game,
        parent_to_children: Dict[Node, Nodes],
        init_pareto_point: ParetoPoint = None,
        graph_name: str = None,
        debug: bool = False):
        super().__init__(game, graph_name)

        self._parent_to_children = parent_to_children
        self._init_pareto_point = init_pareto_point

        if graph_name is None:
            graph_name = 'DeterministicStrategy'

            if init_pareto_point is not None:
                init_pareto_point = tuple(np.around(init_pareto_point, 2))
                p_str = str(tuple(init_pareto_point))\
                    .replace(' ', '')
                graph_name = f'{graph_name}For{p_str}'

            self._graph_name = graph_name

        if self._is_initialized():
            self.construct_graph(debug)

        self.reset()

    @classmethod
    def from_edges(cls, game, edges: List[Tuple[Node, Node]]):

        parent_to_children = defaultdict(lambda: set())

        for edge in edges:
            parent_to_children[edge[0]].add(edge[1])

        return cls(game, parent_to_children)

    @classmethod
    def from_pareto_fronts(cls,
        game,
        pareto_fronts: Dict[Node, Any],
        epsilon: int = 5,
        **kwargs):

        parent_to_children = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        branching_nodes = queue.Queue()

        for u_node, front in pareto_fronts.items():
            for pareto_point in front.pareto_points:
                for v_node in game._graph.successors(u_node):
                    weights = game.get_edge_attributes(u_node, v_node, 'weights')
                    if weights is None:
                        continue
                    weights = np.array(list(weights.values()))

                    p = np.array(pareto_point) - weights
                    # TODO: Is there only one successor pareto that satisfies the parent's pareto?
                    for v_pareto_point in pareto_fronts[v_node].pareto_points:
                        if all(np.around(p, epsilon) >= np.around(v_pareto_point, epsilon)):
                            parent_to_children[u_node][tuple(pareto_point)][v_node] = v_pareto_point

        return cls(game, parent_to_children, **kwargs)

    def _is_initialized(self) -> bool:
        return self._init_pareto_point is not None

    def _check_if_initialized(self):
        if not self._is_initialized():
            raise Exception('Please provide an initial pareto point')

    def set_initial_pareto_point(self, initial_pareto_point: ParetoPoint):
        self._init_pareto_point = init_pareto_point

    def construct_graph(self, debug: bool = False):
        # add this graph object of type of Networkx to our Graph class
        graph = nx.MultiDiGraph(name=self._graph_name)

        init_node = self._game.get_initial_states()[0][0]
        init_attr = self._game._graph.nodes[init_node]
        graph.add_node(init_node, **init_attr)

        accept_node = self._game.get_accepting_states()[0]
        curr_node = init_node

        search_queue = queue.Queue()
        search_queue.put((curr_node, self._init_pareto_point))
        visited = set()
        visited.add(init_node)

        # Until we find all paths for the current pareto point
        # while curr_node != accept_node and not search_queue.empty():
        while not search_queue.empty():
            curr_node, pareto_point = search_queue.get()

            for next_node, next_pareto_point in \
                self._parent_to_children[curr_node][tuple(pareto_point)].items():
                attr = self._game._graph.nodes[next_node]
                edge_attr = self._game._graph[curr_node][next_node][0]
                graph.add_node(next_node, **attr)
                graph.add_edge(curr_node, next_node, **edge_attr)

                if next_node not in visited:
                    visited.add(next_node)
                    search_queue.put((next_node, next_pareto_point))

        # Initialize each node's distance
        min_distance = defaultdict(lambda: np.inf)
        min_distance[accept_node] = 0

        min_distance_prev = None
        visitation_order = self._decide_visitation_order(graph, accept_node)

        # Backward value iteration to find shortest distance for all nodes
        while min_distance != min_distance_prev:

            min_distance_prev = copy.deepcopy(min_distance)

            for curr_node in visitation_order:

                player = graph.nodes[curr_node].get('player')
                ss = list(graph.successors(curr_node))
                successor_distances = [min_distance_prev[s] for s in ss]

                if len(successor_distances) == 0:
                    continue

                if player == 'eve':
                    min_distance[curr_node] = min(successor_distances) + 1
                else:
                    min_distance[curr_node] = max(successor_distances) + 1

        # Forward search for finding a path that always has a successor with <= distance.
        self._graph = nx.MultiDiGraph(name=self._graph_name)
        self.add_state(init_node, **init_attr)

        search_queue = queue.Queue()
        search_queue.put(init_node)
        visited = set(init_node)

        while not search_queue.empty():

            curr_node = search_queue.get()

            for next_node in graph.successors(curr_node):

                if min_distance[curr_node] < min_distance[next_node]:
                    continue

                attr = self._game._graph.nodes[next_node]
                edge_attr = self._game._graph[curr_node][next_node][0]
                self.add_state(next_node, **attr)
                self.add_edge(curr_node, next_node, **edge_attr)

                if next_node not in visited:
                    visited.add(next_node)
                    search_queue.put(next_node)

    def _decide_visitation_order(self, graph, accept_node):
        search_queue = queue.Queue()
        visited = set()
        visitation_order = []

        search_queue.put(accept_node)
        visited.add(accept_node)
        visitation_order.append(accept_node)

        while not search_queue.empty():

            v_node = search_queue.get()

            for u_node in graph.predecessors(v_node):
                # Avoid self loop
                if u_node == v_node:
                    continue

                if u_node not in visited:
                    visitation_order.append(u_node)
                    visited.add(u_node)
                    search_queue.put(u_node)

        visitation_order.remove(accept_node)

        return visitation_order

    def construct_graph_w(self, debug: bool = False):
        # add this graph object of type of Networkx to our Graph class
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        init_node = self._game.get_initial_states()[0][0]
        attr = self._game._graph.nodes[init_node]
        self.add_state(init_node, **attr)

        accept_node = self._game.get_accepting_states()[0]
        curr_node = init_node

        search_queue = queue.Queue()
        search_queue.put((curr_node, self._init_pareto_point))
        nodes = [init_node]
        edges = []
        branches = []
        edges_in_branch = defaultdict(lambda: [])

        # Until we find all paths for the current pareto point
        # while curr_node != accept_node and not search_queue.empty():
        while not search_queue.empty():

            curr_node, pareto_point = search_queue.get()

            player = self._game.get_state_w_attribute(curr_node, 'player')
            paretos = list(self._parent_to_children[curr_node][tuple(pareto_point)].values())
            n_duplicate = paretos.count(pareto_point)
            i_duplicate = 0

            for next_node, next_pareto_point in \
                self._parent_to_children[curr_node][tuple(pareto_point)].items():

                # Check if there is a loop. If so, delete edges until the most recent branch
                if next_node in nodes and len(branches) != 0 and \
                    all(pareto_point == next_pareto_point):

                    branch_node, other_next_node = branches.pop()
                    # Search for other branch.
                    # It must have the exact pareto point as curr_node if it's a loop
                    edges.append((branch_node, other_next_node))
                    search_queue.put((other_next_node, pareto_point))

                    # Delete all nodes and edges
                    nodes.remove(edges_in_branch[branch_node][0][0])
                    for edge in edges_in_branch[branch_node]:
                        edges.remove(edge)
                        nodes.remove(edge[1])
                    del edges_in_branch[branch_node]

                    continue

                # If sys has multiple outgoing edges with same pareto points,
                # it is likely that it has a loop. Corner cases exist.
                if player == 'eve' and n_duplicate > 1 and pareto_point == next_pareto_point:
                    i_duplicate += 1
                    if i_duplicate > 1:
                        branches.append((curr_node, next_node))
                        continue
                    edges_in_branch[curr_node].append((curr_node, next_node))

                edges.append((curr_node, next_node))
                nodes.append(next_node)
                search_queue.put((next_node, next_pareto_point))

        # Construct a graph
        init_node = self._game.get_initial_states()[0][0]
        attr = self._game._graph.nodes[init_node]
        self.add_state(init_node, **attr)

        for u_node, v_node in edges:

            attr = self._game._graph.nodes[v_node]
            edge_attr = self._game._graph[u_node][v_node][0]

            self.add_state(v_node, **attr)
            self.add_edge(u_node, v_node, **edge_attr)

    def construct_graph__(self, debug: bool = False):
        # add this graph object of type of Networkx to our Graph class
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        init_node = self._game.get_initial_states()[0][0]
        attr = self._game._graph.nodes[init_node]
        self.add_state(init_node, **attr)

        accept_node = self._game.get_accepting_states()[0]
        curr_node = init_node

        search_queue = queue.Queue()
        search_queue.put((curr_node, self._init_pareto_point))
        visited = set()
        visited.add(init_node)

        # 1. Generate a strategy graph
        while not search_queue.empty():
            curr_node, pareto_point = search_queue.get()

            for next_node, next_pareto_point in \
                self._parent_to_children[curr_node][tuple(pareto_point)].items():
                attr = self._game._graph.nodes[next_node]
                edge_attr = self._game._graph[curr_node][next_node][0]
                self.add_state(next_node, **attr)
                self.add_edge(curr_node, next_node, **edge_attr)

                if next_node not in visited:
                    visited.add(next_node)
                    search_queue.put((next_node, next_pareto_point))

        # 2. Run a backward reachability analysis
        search_queue = queue.Queue()
        search_queue.put(accept_node)
        visited = set()
        visited.add(accept_node)
        regions = defaultdict(lambda: -1)

        while not search_queue.empty():
            curr_node = search_queue.get()
            for prev_node in self._graph.predecessors(curr_node):
                if prev_node not in visited:
                    visited.add(prev_node)
                    search_queue.put(prev_node)

        # 3. Delete loops
        for node in self._graph.nodes():
            if node not in visited:
                print(node)

    def fancy_graph(self, color=("lightgrey", "red", "purple"), **kwargs) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1]})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2]})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "circle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = f"a={edge[2].get('actions')}"
            dot.edge(str(edge[0]), str(edge[1]), label=label)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, **kwargs)

    def reset(self):
        virtual_init_node = self.get_initial_states()[0][0]
        self.curr_state = list(self._graph.successors(virtual_init_node))[0]
        return None

    def step(self, env_action: str):
        accepted = False

        if env_action is not None:
            for next_state in self._graph.successors(self.curr_state):
                actions = self.get_edge_attributes(self.curr_state, next_state, 'actions')
                if env_action in actions:
                    self.curr_state = next_state
                    break
        # Now we should be at the system's state
        if self.curr_state is self.get_accepting_states()[0][0]:
            accepted = True

        successors = list(self._graph.successors(self.curr_state))
        index = random.randint(0, len(successors)-1)
        next_state = successors[index]
        actions = self.get_edge_attributes(self.curr_state, next_state, 'actions')
        self.curr_state = next_state
        # Now we should be at the environment's state

        accepting_states = self.get_accepting_states()
        if next_state in accepting_states:
            accepted = True

        return list(actions)[0], accepted


class StochasticStrategy(Strategy):
    """
    init_dist must be set
    """
    def __init__(self,
        game,
        update_state_dict: Dict[Node, Dict[ParetoPoint, Dict[Action, DistOverPareto]]],
        update_move_dict: Dict[Node, Dict[Action, Dict[ParetoPoint, Node]]],
        init_pareto_point: Point = None,
        init_dist: DistOverPareto = None,
        graph_name: str = None,
        epsilon: int = 5):
        super().__init__(game, graph_name)

        self._update_state_dict = update_state_dict
        self._update_move_dict = update_move_dict
        self._epsilon = epsilon

        self._init_dist = init_dist

        if graph_name is None:
            graph_name = 'StochasticStrategy'

            if init_pareto_point is not None:
                self.set_init_point(init_pareto_point)
                init_pareto_point = tuple(np.around(init_pareto_point, 2))
                p_str = str(tuple(init_pareto_point))\
                    .replace(' ', '')
                graph_name = f'{graph_name}For{p_str}'

            self._graph_name = graph_name

        if self._is_initialized():
            self.construct_graph()

        self._curr_node = VIRTUAL_INIT
        self._curr_pareto = 0

    @classmethod
    def from_flat_dict(
        cls,
        game,
        flat_update_state_dict: Dict[Tuple[Node, ParetoPoint, Action, ParetoPoint], Probability],
        flat_update_move_dict: Dict[Tuple[Node, Action, ParetoPoint], Node],
        **kwargs):

        update_state_dict: Dict[Node, Dict[ParetoPoint, Dict[Action, DistOverPareto]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))

        for (n, cp, a, np), prob in flat_update_state_dict.items():
            update_state_dict[n][cp][a][np] = prob

        update_move_dict: Dict[Node, Dict[Action, Dict[ParetoPoint, Node]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

        for (u_n, a, p), v_n in flat_update_move_dict.items():
            update_move_dict[u_n][a][p] = v_n

        return cls(game, update_state_dict, update_move_dict, **kwargs)

    @classmethod
    def from_pareto_fronts(
        cls,
        game,
        pareto_fronts: Dict[Node, Any],
        **kwargs):

        update_state_dict: Dict[Node, Dict[ParetoPoint, Dict[Action, DistOverPareto]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))
        update_move_dict: Dict[Node, Dict[Action, Dict[ParetoPoint, Node]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

        for u_node, front in pareto_fronts.items():
            for curr_pareto_point in front.pareto_points:
                for v_node in game._graph.successors(u_node):

                    if not pareto_fronts[v_node].is_initialized():
                        continue

                    action = game.get_edge_attributes(u_node, v_node, 'actions')
                    weights = game.get_edge_attributes(u_node, v_node, 'weights')
                    next_pareto_points = pareto_fronts[v_node].pareto_points

                    # Main: Compute the memory transition probabilities
                    probs = solve_memory_trans_probabilities(
                        curr_pareto_point=curr_pareto_point,
                        edge_weight=list(weights.values()),
                        next_pareto_points=next_pareto_points)

                    if probs is None:
                        continue

                    # Update MemoryStateDict
                    for pp, prob in zip(next_pareto_points, probs):
                        update_state_dict[u_node][tuple(curr_pareto_point)][action]\
                            [tuple(pp)] = prob

                    # Update MemoryMoveDict
                    for pp in next_pareto_points:
                        update_move_dict[u_node][action][tuple(pp)] = v_node

        stochastic_strategy = cls(game, update_state_dict, update_move_dict, **kwargs)
        return stochastic_strategy

    def _init_dist_from_init_pareto_point(self, init_pareto_point) -> DistOverPareto:
        init_node = self._game.get_initial_states()[0][0]
        next_pareto_points = list(self._update_state_dict[init_node].keys())

        probs = solve_memory_trans_probabilities(
            curr_pareto_point=init_pareto_point,
            edge_weight=np.zeros(len(init_pareto_point)),
            next_pareto_points=next_pareto_points)

        return {tuple(pp): prob for pp, prob in zip(next_pareto_points, probs)}

    def _is_initialized(self) -> bool:
        return self._init_dist is not None

    def _check_if_initialized(self):
        if not self._is_initialized():
            raise Exception('Please provide the initial distribution over the pareto points')

    def set_init_point(self, init_pareto_point: Point):
        init_dist = self._init_dist_from_init_pareto_point(init_pareto_point)
        self.set_initial_distribution(init_dist)

    def set_initial_distribution(self, init_dist: DistOverPareto):
        self._init_dist = init_dist

    def construct_graph(self):

        self._check_if_initialized()

        # add this graph object of type of Networkx to our Graph class
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        init_node = self._game.get_initial_states()[0][0]

        # Create an virtual initial state called VIRTUAL_INIT
        # Then, create an edge from INIT to the actual initial states
        for pareto, prob in self._init_dist.items():

            prob = np.around(prob, self._epsilon)
            if prob == 0.0:
                continue

            self._add_product_edge(
                state_src=VIRTUAL_INIT, pareto_src=0,
                state_dest=init_node, pareto_dest=pareto,
                action=None,
                probability=prob)

        # Construct the rest of the graph
        for u_node, attr1 in self._update_state_dict.items():
            for u_pareto, attr2 in attr1.items():
                for action, dist in attr2.items():
                    for v_pareto, prob in dist.items():

                        v_node = self._update_move_dict[u_node][action][v_pareto]

                        prob = np.around(prob, self._epsilon)
                        if prob == 0.0:
                            continue

                        self._add_product_edge(
                            state_src=u_node, pareto_src=u_pareto,
                            state_dest=v_node, pareto_dest=v_pareto,
                            action=action,
                            probability=prob)

    def _add_product_edge(self,
        state_src: Node, pareto_src: ParetoPoint,
        state_dest: Node, pareto_dest: ParetoPoint,
        **kwargs) -> None:

        product_src = self._add_product_node(state_src, pareto_src)
        product_dest = self._add_product_node(state_dest, pareto_dest)

        self.add_edge(product_src, product_dest, **kwargs)

    def _add_product_node(self, state: Node, pareto: ParetoPoint) -> None:

        product_state = (state, pareto)

        attr = {}
        if state == VIRTUAL_INIT:
            attr = {'init': True}
        else:
            attr = copy.deepcopy(self._game._graph.nodes[state])
            if 'init' in attr:
                attr['init'] = False

        self.add_state(product_state, **attr)

        return product_state

    def fancy_graph(self, color=("lightgrey", "red", "purple"), **kwargs) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            node_name = (n[0][0], tuple(np.around(n[0][1], 2)))
            dot.node(str(node_name), _attributes={"style": "filled",
                                                  "fillcolor": color[0],
                                                  "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1]})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2]})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "circle"})
            if n[1].get('player') == 'adam':
                dot.node(str(n[0]), _attributes={"shape": "rectangle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = f"a={edge[2].get('action')}, p={edge[2].get('probability')}"
            dot.edge(str(edge[0]), str(edge[1]), label=label)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, **kwargs)

    def available_actions(self, next_state: Node) -> Actions:

        self._check_if_initialized()

        if next_state not in self._game._graph.successors(self._curr_state):
            raise Exception(f'{next_state} is not in the successors of {self._curr_state}')

        pass

    def step(self, state: Node):
        pass

    def get_edges_on_original_graph(self):
        edges = self._graph.edges()

        edges_on_original = []
        for edge in edges:
            if edge[0][0] == VIRTUAL_INIT:
                continue
            u_node = edge[0][0]
            v_node = edge[1][0]
            edges_on_original.append((u_node, v_node))

        return edges_on_original


# TODO: These won't be needed
class MealyMachineBase(Graph):
    # TODO: We only need a graph and a pareto front at each node

    """Initial State: s \in S"""
    _init_state: int = None

    """Initial Distribution: S -> D(M)"""
    _init_dist: rv_discrete = None          # TODO: This can be set later but it must be provided.

    """Next Function: S x M -> A"""
    #               state, memory, action
    _next_dict: Dict[int, Dict[int, int]] = None        # TODO: Not Needed

    """Memory Update States Function: S x M x A -> D(M)"""
    #                       state, memory, action, distribution(memory)
    _update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None

    """Memory Update Moves Function: S x A x M -> S x D(M)"""
    #                       state, action, memory, next_state, distribution(memory)
    _update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None

    """Current State s"""
    _curr_state: int = None

    """Current Action a"""
    _curr_action: int = None

    """Current Memory m"""
    _curr_memory: int = None

    """On-Move Memory o_m"""
    _onmove_memory: int = None

    def __init__(self,
        init_state: int = None,
        init_distribution: rv_discrete = None,
        next_dict: Dict[int, Dict[int, int]] = None,
        update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None,
        update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None,
        graph_name: str = 'MooreMachine',
        epsilon: int = 5):
        Graph.__init__(self, config_yaml=None, save_flag=True)

        self._init_state = init_state
        self._init_distribution = init_distribution
        self._next_dict = next_dict
        self._update_state_dict = update_state_dict
        self._update_move_dict = update_move_dict
        self._graph_name = graph_name
        self._epsilon = epsilon

        self._check_initialized()

        self._curr_state = self._init_state
        self._curr_memory = self._init_distribution.rvs(size=1)[0]
        self.construct_graph()

    def construct_graph(self):
        # add this graph object of type of Networkx to our Graph class
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        for i_p, prob in zip(self._init_distribution.xk, self._init_distribution.pk):
            prob = np.around(prob, self._epsilon)
            self._add_product_edge(
                state_src=VIRTUAL_INIT, memory_src=0,
                state_dest=self._init_state, memory_dest=i_p,
                action=None,
                probability=prob)

        for u_node_idx, attr1 in self._update_state_dict.items():
            for u_pareto_idx, attr2 in attr1.items():
                for action_idx, dist in attr2.items():
                    for v_pareto_idx, prob in zip(dist.xk, dist.pk):

                        prob = np.around(prob, self._epsilon)

                        for v_node_idx in \
                            self._update_move_dict[u_node_idx][action_idx][v_pareto_idx].keys():


                            self._add_product_edge(
                                state_src=u_node_idx, memory_src=u_pareto_idx,
                                state_dest=v_node_idx, memory_dest=v_pareto_idx,
                                action=action_idx,
                                probability=prob)

    def _add_product_edge(self,
        state_src, memory_src,
        state_dest, memory_dest,
        **kwargs):

        product_src = self._add_product_node(state_src, memory_src)
        product_dest = self._add_product_node(state_dest, memory_dest)
        self.add_edge(product_src, product_dest, **kwargs)

    def _add_product_node(self, state, memory):
        product_state = self._get_product_state_label(state, memory)
        if state == VIRTUAL_INIT:
            self.add_state(product_state, init=True)
        else:
            self.add_state(product_state)

        return product_state

    def _get_product_state_label(self, state, memory):
        """
        Computes the combined product state label

        :param      dynamical_system_state:  The dynamical system state label
        :param      specification_state:     The specification state label

        :returns:   The product state label.
        """

        if type(state) != str:
            state = str(state)

        if type(memory) != str:
            memory = str(memory)

        return state + ', ' + memory

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            dot.node(str(n[0]), _attributes={"style": "filled",
                                             "fillcolor": color[0],
                                             "shape": "rectangle"})
            if n[1].get('init'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1]})
            if n[1].get('accepting'):
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2]})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = f"a={edge[2].get('action')}, p={edge[2].get('probability')}"
            dot.edge(str(edge[0]), str(edge[1]), label=label)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            self.save_dot_graph(dot, self._graph_name, True)

    def get_memories(self, curr_state: int) -> List[int]:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            curr_memories = set(self._update_state_dict[curr_state].keys())
            return curr_memories

        raise ValueError(f'Invalid state and memory ({curr_state})')

    def take_action(self, curr_state: int = None, curr_memory: int = None) -> int:
        """Take an aciton and return the next state"""
        if curr_state is None:
            curr_state = self._curr_state

        if curr_memory is None:
            curr_memory = self._curr_memory

        # First, Choose an action
        curr_action = self.sample_action(curr_state, curr_memory)

        # Update Memory
        onmove_memory = self.sample_on_move_memory(curr_state, curr_memory, curr_action)

        # Take a transition
        next_state = self.get_next_states(curr_state, curr_action, onmove_memory)
        next_memory = self.sample_next_memory(curr_state, curr_action, onmove_memory)

        self._curr_state = next_state
        self._curr_memory = next_memory

        return next_state

    def sample_action(self, curr_state: int, curr_memory: int) -> int:
        return self._sample_action(curr_state, curr_memory)

    def __sample_action(self, curr_state: int, curr_memory: int) -> int:
        actions = self._get_actions(curr_state, curr_memory)

        if isinstance(actions, List):
            # Adam, Env
            return random.choice(actions)
        else:
            # Eve, Sys
            return actions

    def get_actions(self, curr_state: int, curr_memory: int) -> List[int]:
        return self.__get_actions(curr_state, curr_memory)

    def __get_actions(self, curr_state: int, curr_memory: int) -> List[int]:

        self._check_initialized()

        # Eve's State
        if curr_state in self._next_dict:
            if curr_memory in self._next_dict[curr_state]:
                return [self._next_dict[curr_state][curr_memory].rvs(size=1)[0]]
            else:
                raise ValueError(f'Memory {curr_memory} is not in the next function')

        # Adam's State
        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                next_actions = set(self._update_state_dict[curr_state][curr_memory].keys())
                return next_actions

        raise ValueError(f'Invalid state and memory ({curr_state}, {curr_memory})')

    def sample_on_move_memory(self, curr_state: int, curr_memory: int, curr_action: int) -> int:
        return self.__sample_on_move_memory(curr_state, curr_memory, curr_action)

    def __sample_on_move_memory(self, curr_state: int, curr_memory: int, curr_action: int) -> int:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                if curr_action in self._update_state_dict[curr_state][curr_memory]:
                    return self._update_state_dict[curr_state][curr_memory].rvs(sample=1)[0]

        msg = f'Invalid state, memory, action = ({curr_state}, {curr_memory}, {curr_action})'
        raise ValueError(msg)

    def get_onmove_memories(self, curr_state: int, curr_memory: int, curr_action: int) -> List[int]:
        return self.__get_onmove_memories()

    def __get_onmove_memories(self, curr_state: int, curr_memory: int, curr_action: int) -> List[int]:

        self._check_initialized()

        if curr_state in self._update_state_dict:
            if curr_memory in self._update_state_dict[curr_state]:
                if curr_action in self._update_state_dict[curr_state][curr_memory]:
                    return self._update_state_dict[curr_state][curr_memory][curr_action].xk

        msg = f'Invalid state, memory, action = ({curr_state}, {curr_memory}, {curr_action})'
        raise ValueError(msg)

    # TODO: Currently, we assume next state is deterministically transitioned from s,a,m
    def get_next_state(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:
        return self.__get_next_state(curr_state, curr_action, onmove_memory)

    def __get_next_state(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:

        self._check_initialized()

        if curr_state in self._update_move_dict:
            if curr_action in self._update_move_dict[curr_state]:
                if onmove_memory in self._update_move_dict[curr_state][curr_action]:
                    next_state = list(self._update_move_dict[curr_state][curr_action][onmove_memory].keys())[0]
                    return next_state

        msg = f'Invalid state, action, memory ({curr_state}, {curr_action}, {onmove_memory})'
        raise ValueError(msg)

    def sample_next_memory(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:
        return self.__sample_next_memory(curr_state, curr_action, onmove_memory)

    def __sample_next_memory(self, curr_state: int, curr_action: int, onmove_memory: int) -> int:

        self._check_initialized()

        next_state = self.__get_next_state(curr_state, curr_action, onmove_memory)

        if next_state in self._update_move_dict[curr_state][curr_action][onmove_memory]:
            return self._update_move_dict[curr_state][curr_action][onmove_memory][next_state].rvs(sample=1)[0]

        msg = f'Invalid state, action, memory ({curr_state}, {curr_action}, {onmove_memory})'
        raise ValueError(msg)

    def get_next_memories(self, curr_state: int, curr_action: int,
                          onmove_memory: int, next_state: int) -> List[int]:
        return self.__get_next_memories(curr_state, curr_action, onmove_memory, next_state)

    def __get_next_memories(self, curr_state: int, curr_action: int,
                          onmove_memory: int, next_state: int) -> List[int]:

        self._check_initialized()

        next_state = self.__get_next_state(curr_state, curr_action, onmove_memory)

        if next_state in self._update_move_dict[curr_state][curr_action][onmove_memory]:
            next_memories = self._update_move_dict[curr_state][curr_action][onmove_memory][next_state].xk
            return next_memories

        msg = f'Invalid state, action, memory, next_state ({curr_state}, {curr_action}, {onmove_memory}, {next_state})'
        raise ValueError(msg)

    def get_transitions(self) -> Dict[int, List[int]]:

        states = set(self._update_move_dict.keys())
        transitions: Dict[int, List[int]] = defaultdict(lambda: [])

        for state in states:

            memories = self.get_memories(state)
            for memory in memories:

                actions = self.__get_actions(state, memory)
                for action in actions:

                    onmove_memories = self.__get_onmove_memories(state, memory, action)
                    for onmove_memory in onmove_memories:

                        next_state = self.__get_next_state(state, action, onmove_memory)
                        transition = {'next_node': next_state, 'action': action}

                        transitions[state].append(transition)

        return transitions

    def _check_initialized(self):
        if not self.initialized:
            raise Exception('Not initialized')

    @property
    def initialized(self):
        if None in [self._init_state,
            self._init_distribution,
            self._next_dict,
            self._update_state_dict,
            self._update_move_dict]:
            return False
        return True

    @property
    def init_state(self) -> int:
        self._check_initialized()
        return self._init_state

    @property
    def curr_state(self) -> int:
        return self._curr_state

    @property
    def curr_memory(self) -> int:
        return self._curr_memory

    @property
    def curr_action(self) -> int:
        if self._curr_action is None:
            raise Exception('Current Action not initialized. Please take an action first')
        return self._curr_action


class MealyMachine(MealyMachineBase):

    """Node Index to Node Name"""
    idx_to_state: Dict[int, Node] = None

    """Action Index to Action Name"""
    _idx_to_action: Dict[int, Dict[int, Action]] = None

    def __init__(self,
        idx_to_state: Dict[int, Node] = None,
        idx_to_action: Dict[int, Dict[int, Action]] = None,
        init_state: int = None,
        init_distribution: rv_discrete = None,
        next_dict: Dict[int, Dict[int, int]] = None,
        update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None,
        update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None,
        **kwargs):
        self._idx_to_state = idx_to_state
        self._idx_to_action = idx_to_action
        super().__init__(init_state, init_distribution, next_dict,
            update_state_dict, update_move_dict, **kwargs)

        self._check_initialized()

        self._state_to_idx = dict(zip(self._idx_to_state.values(),
                                      self._idx_to_state.keys()))

    def init_state(self) -> Node:
        state_idx = super().init_state()
        return self._idx_to_state[state_idx]

    @property
    def curr_state(self) -> Node:
        return self._idx_to_state[self._curr_state]

    @property
    def curr_action(self) -> Action:
        if self._curr_action is None:
            raise Exception('Current Action not initialized. Please take an action first')

        return self._idx_to_action[self.curr_state][self._curr_action]

    def sample_action(self, curr_state: Node, curr_memory: int) -> Action:
        # Assume curr_state is of type Node
        state_idx = self._state_to_idx(curr_state)

        action_idx = super().sample_action(state_idx, curr_memory)
        return self._idx_to_action[state_idx][action_idx]

    def get_actions(self, curr_state: Node, curr_memory: int) -> List[Action]:
        state_idx = self._state_to_idx(curr_state)

        action_idxs = super().get_actions(state_idx, curr_memory)
        return [self._idx_to_action[state_idx][action_idx] for action_idx in action_idxs]

    def get_next_state(self, curr_state: Node, curr_action: Action,
                       onmove_memory: int) -> Node:
        state_idx = self._state_to_idx(curr_state)

        action_idx = self._idx_to_action[state_idx].index(curr_action)

        state_idx = super().get_next_state(state_idx, action_idx, onmove_memory)
        return self._idx_to_state[state_idx]

    def get_transitions(self, return_dict: bool = True, return_edges: bool = False) \
        -> Union[Dict[int, List[Dict[Node, Node]]], List[Dict[Node, Node]]]:

        if sum([return_dict, return_edges]) != 1:
            raise ValueError('Select either "return_dict" or "return_edges"')

        transitions = super().get_transitions()

        if return_dict:
            game_transitions = defaultdict(lambda: [])

            for curr_state_idx, transition in transitions.items():
                for transition in transition:
                    next_state_idx = transition['next_node']
                    sta_action = transition['action']
                    curr_state = self._idx_to_state[curr_state_idx]
                    next_state = self._idx_to_state[next_state_idx]
                    action = self._idx_to_action[curr_state_idx][sta_action]

                    transition = {'next_node': next_state, 'action': action}
                    game_transitions[curr_state].append(transition)

            return game_transitions

        elif return_edges:
            game_transitions = []

            for curr_state_idx, next_states_idx in transitions.items():
                for next_state_idx in next_states_idx:
                    curr_state = self._sta_to_game(curr_state_idx)
                    next_state = self._sta_to_game(next_state_idx)
                    game_transitions.append((next_state, next_state))

            return game_transitions

    def construct_graph(self):
        # add this graph object of type of Networkx to our Graph class
        self._graph = nx.MultiDiGraph(name=self._graph_name)

        # Create an virtual initial state called INIT
        # Then, create an edge from INIT to the actual initial states
        for i_p, prob in zip(self._init_distribution.xk, self._init_distribution.pk):

            prob = np.around(prob, self._epsilon)
            if prob == 0.0:
                continue

            self._add_product_edge(
                state_src=INIT, memory_src=0,
                state_dest=self._init_state, memory_dest=i_p,
                action=None,
                probability=prob)

        # Construct the rest of the graph
        for u_node_idx, attr1 in self._update_state_dict.items():
            for u_pareto_idx, attr2 in attr1.items():
                for action_idx, dist in attr2.items():
                    for v_pareto_idx, prob in zip(dist.xk, dist.pk):

                        for v_node_idx in \
                            self._update_move_dict[u_node_idx][action_idx][v_pareto_idx].keys():

                            u_node = self._idx_to_state[u_node_idx]
                            v_node = self._idx_to_state[v_node_idx]
                            action = self._idx_to_action[u_node_idx][action_idx]
                            prob = np.around(prob, self._epsilon)
                            if prob == 0.0:
                                continue

                            self._add_product_edge(
                                state_src=u_node, memory_src=u_pareto_idx,
                                state_dest=v_node, memory_dest=v_pareto_idx,
                                action=action,
                                probability=prob)

    def _add_product_node(self, state, memory):
        product_state = self._get_product_state_label(state, memory)

        kwargs = {}

        if state == INIT:
            kwargs['init'] = True
        elif state == ACCEPT:
            kwargs['accepting'] = True

        self.add_state(product_state, **kwargs)

        return product_state


class PrismMealyMachine(MealyMachineBase):

    """Prism STA state (strategy state) to prism model state"""
    _sta_to_prism_state_map: Dict[int, int] = None

    """Prism model state to game state"""
    _prism_to_game_map: Dict[int, Node] = None

    """State & Action idx to Action String"""
    _prism_action_idx_to_game_map: Dict[int, Dict[int, Action]] = None

    def __init__(self,
        sta_to_prism_state_map: Dict[int, int] = None,
        prism_to_game_map: Dict[int, Node] = None,
        prism_action_idx_to_game_map: Dict[int, Dict[int, Action]] = None,
        init_state: int = None,
        init_distribution: rv_discrete = None,
        next_dict: Dict[int, Dict[int, int]] = None,
        update_state_dict: Dict[int, Dict[int, Dict[int, rv_discrete]]] = None,
        update_move_dict: Dict[int, Dict[int, Dict[int, Dict[int, rv_discrete]]]] = None):
        super().__init__(init_state, init_distribution, next_dict,
            update_state_dict, update_move_dict)

        self._sta_to_prism_state_map = sta_to_prism_state_map
        self._prism_to_game_map = prism_to_game_map
        self._prism_action_idx_to_game_map = prism_action_idx_to_game_map

        self._check_initialized()

        self._prism_to_sta_state_map = dict(zip(self._sta_to_prism_state_map.values(),
                                                self._sta_to_prism_state_map.keys()))
        self._game_to_prism_map = dict(zip(self._prism_to_game_map.values(),
                                           self._prism_to_game_map.keys()))
        # self._game_to_prism_action_idx_map = dict(zip(self._prism_action_idx_to_game_map.values(),
        #                                               self._prism_action_idx_to_game_map.keys()))

    def _sta_to_game(self, sta_state: int) -> Node:
        prism_state = self._sta_to_prism_state_map[sta_state]
        return self._prism_to_game_map[prism_state]

    def _game_to_sta(self, game_state: Node) -> int:
        prism_state = self._game_to_prism_map[game_state]
        return self._prism_to_sta_state_map[prism_state]

    def init_state(self) -> Node:
        state_idx_sta = super().init_state()
        return self._sta_to_game(state_idx_sta)

    def get_transitions(self, return_dict: bool = True, return_edges: bool = False) \
        -> Union[Dict[int, List[Dict[Node, Node]]], List[Dict[Node, Node]]]:

        if sum([return_dict, return_edges]) != 1:
            raise ValueError('Select either "return_dict" or "return_edges"')

        transitions = super().get_transitions()

        if return_dict:
            game_transitions = defaultdict(lambda: [])

            for sta_curr_state, transition in transitions.items():
                for transition in transition:
                    sta_next_state = transition['next_node']
                    sta_action = transition['action']
                    game_curr_state = self._sta_to_game(sta_curr_state)
                    game_next_state = self._sta_to_game(sta_next_state)
                    prism_action = self._prism_action_idx_to_game_map[sta_curr_state][sta_action]

                    transition = {'next_node': game_next_state, 'action': prism_action}
                    game_transitions[game_curr_state].append(transition)

            return game_transitions

        elif return_edges:
            game_transitions = []

            for sta_curr_state, sta_next_states in transitions.items():
                for sta_next_state in sta_next_states:
                    game_curr_state = self._sta_to_game(sta_curr_state)
                    game_next_state = self._sta_to_game(sta_next_state)
                    game_transitions.append((game_next_state, game_next_state))

            return game_transitions

    @property
    def curr_state(self) -> Node:
        return self._sta_to_game(self._curr_state)

    @property
    def curr_action(self) -> Action:
        if self._curr_action is None:
            raise Exception('Current Action not initialized. Please take an action first')

        return self._prism_action_idx_to_game_map[self.curr_state][self._curr_action]

    def sample_action(self, curr_state: Node, curr_memory: int) -> Action:
        # Assume curr_state is of type Node
        sta_curr_state = self._game_to_sta(curr_state)

        action_idx = super().sample_action(sta_curr_state, curr_memory)
        return self._prism_action_idx_to_game_map[sta_curr_state][action_idx]

    def get_actions(self, curr_state: Node, curr_memory: int) -> List[Action]:
        sta_curr_state = self._game_to_sta(curr_state)

        action_idxs = super().get_actions(sta_curr_state, curr_memory)
        return [self._prism_action_idx_to_game_map[sta_curr_state][action_idx] for action_idx in action_idxs]

    def get_next_state(self, curr_state: Node, curr_action: Action,
                       onmove_memory: int) -> Node:
        sta_curr_state = self._game_to_sta(curr_state)

        sta_curr_action = self._prism_action_idx_to_game_map[sta_curr_state].index(curr_action)

        state_idx = super().get_next_state(sta_curr_state, sta_curr_action, onmove_memory)
        return self._sta_to_game(state_idx)
