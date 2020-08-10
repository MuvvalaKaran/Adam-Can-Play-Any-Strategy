# class that implements cumulative payoff
import copy
import math
import warnings
import sys

from collections import defaultdict
from typing import Dict, Tuple, List

# import local packages
from src.factory.builder import Builder
from .base import Payoff
from src.graph import Graph


class FinitePayoff(Payoff):

    def __init__(self,
                 graph: Graph,
                 payoff: callable) -> 'FinitePayoff()':

        Payoff.__init__(self,
                        graph=graph,
                        payoff=payoff)

    def cycle_main(self):
        """
        A method to compute the all the possible loop values for a given graph with an init node. Before calling this
        function make sure you have already updated the init node and then call this function.
        :return: A dict consisting all the loops that exist in a graph with a given init node and its corresponding
        values for a given payoff function
        """
        # NOTE: the data player and the init flag cannot be accessed as graph[node]['init'/ 'player'] you have to first
        #  access the data as graph.nodes.data() and loop over the list each element in that list is a tuple
        #  (NOT A DICT) of the form (node_name, {key: value})

        # create a dict to hold values of the loops as str for a corresponding play which is a str too
        loop_dict: Dict[Tuple, float] = {}

        # visit each neighbour of the init node
        for node in self.graph._graph.neighbors(self.init_node):
            visitStack: Dict[Tuple, bool] = defaultdict(lambda: False)
            visitStack[self.init_node] = True
            nodeStack: List[Tuple] = []
            nodeStack.append(self.init_node)

            # check if the init node and the current node are same then directly compute the loop value and
            # store the self play in the loop dict
            if node == self.init_node:
                nodeStack.append(node)
                loop_value: float = self._compute_loop_value(nodeStack)
                self.create_tuple_mapping(tuple(nodeStack))
                loop_dict.update({self.map_tuple_idx[tuple(nodeStack)]: loop_value})
                continue

            self._cycle_util(node, visitStack, loop_dict, nodeStack)

        self._loop_vals = loop_dict

    def _cycle_util(self, node, visitStack: Dict[Tuple, bool], loop_dict: Dict[Tuple, float],
                    nodeStack: List[Tuple]) -> None:
        """
       A method to help with detecting loop and updating the loop dict accordingly.
       :param node: Tuple which is a node that belong to the self.graph
       :param visitStack: A dict that keeps track of all the nodes visited and updates the flag to True if so
       :param loop_dict: A dict that holds values to all possible loops that can be computed for a given graph and for
       a given payoff function
       :param nodeStack: A list that holds all the nodes we visit along a play (nodes can and do repeat in this
       "stack")
       :return: A dict @loop_dict that holds the values of all the possible loops that can be computed for a given
       payoff function @ self.__payoff_func.
       """
        visitStack = copy.copy(visitStack)
        visitStack[node] = True
        nodeStack = copy.copy((nodeStack))
        nodeStack.append(node)
        for neighbour in self.graph._graph.neighbors(node):
            if visitStack[neighbour]:
                nodeStack = copy.copy((nodeStack))
                # for a state with a self-loop, when we start from the same state, it then adds the state thrice.
                # we would like to avoid that using this flag
                if nodeStack.count(neighbour) != 2:
                    nodeStack.append(neighbour)
                # a different logic to cumulative payoff as it is defined for finite traces unlike other payoffs
                # that are defined over infinite traces
                loop_value: float = self._compute_loop_value(nodeStack)
                self.create_tuple_mapping(tuple(nodeStack))
                loop_dict.update({self.map_tuple_idx[tuple(nodeStack)]: loop_value})
                nodeStack.pop()
                continue
            else:
                self._cycle_util(neighbour, visitStack, loop_dict, nodeStack)

    def _compute_loop_value(self, stack: List) -> float:
        """
        A helper method to compute the value of a loop
        :param stack: a List of nodes (of type tuple)
        :return: The value associate with a play (nodes in stack) given a payoff function
        """

        # a container to hold all the edges
        loop_edge_w = []

        # if its a self-loop then consider it once only
        if stack[-1] == stack[-2]:
            for ix in range(0, len(stack) - 1):
                loop_edge_w.append(float(self.graph._graph[stack[ix]][stack[ix + 1]][0].get('weight')))

            return self.payoff_func(loop_edge_w)

        # if its a loop between two nodes then we need to check if all the weights in the loop sum up to be zero.
        # if not we assign the loop a value of positive infinity
        else:
            initial_index = stack.index(stack[-1])
            # get the edge weights between the repeated nodes
            for i in range(initial_index, len(stack) - 1):
                loop_edge_w.append(float(self.graph._graph[stack[i]][stack[i + 1]][0].get('weight')))

            if sum(loop_edge_w) == 0:
                for ix in range(0, initial_index):
                    loop_edge_w.append(float(self.graph._graph[stack[ix]][stack[ix + 1]][0].get('weight')))
                return self.payoff_func(loop_edge_w)

            else:
                # its not a well defined loop - sum of edges is + infinity or -infinity depending on the edge weights.
                if sum(loop_edge_w) > 0:
                    return math.inf
                else:
                    return -1 * math.inf

    def compute_cVal(self, vertex: Tuple, debug: bool = False) -> Tuple[str, List]:
        """
        A Method to compute the cVal using  @vertex as the starting node
        :param vertex: a valid node of the graph
        :param debug: flag to print to print the max cooperative value from a given vertex
        :return: a single value of the max payoff when both adam and eve play cooperatively
        """
        # find all plays in which the vertex exist
        play_dict = {k: v for k, v in self._loop_vals.items() if self._find_vertex_in_play(vertex, k)}

        if play_dict:
            # find the max of the value
            max_play = max(play_dict, key=lambda key: play_dict[key])
            if debug:
                print(f"The cVal from the node {vertex}")
                print(f"for the play {max_play} is {play_dict[max_play]}")
        else:
            warnings.warn(f"There does not exists any loops from the node {vertex}. Please check if the "
                          f"graph is total and that {vertex} has at-least one outgoing edge. "
                          f"If this is true and you still see this error then who knows what's wrong! \n")
            sys.exit(-1)

        return play_dict[max_play], self.map_tuple_idx.inverse[max_play]


class FinitePayoffBuilder(Builder):

    def __init__(self):
        # call to the parent class init to initialize _instance attr to None
        Builder.__init__(self)

    def __call__(self,
                 graph: Graph,
                 payoff_string: str):
        """
        Returns a concrete instance of Payoff class that has the ability to compute the cVal from a given node
        :param graph:
        :param payoff_string:
        :return:
        """

        payoff_val = self._get_payoff_val(payoff_string)

        print(f"Using payoff function : {payoff_string}")

        self._instance = FinitePayoff(graph=graph,
                                payoff=payoff_val)

        return self._instance

    def _get_payoff_val(self, payoff_str: str):

        payoff_dict = {
            'cumulative': sum
        }

        try:
            return payoff_dict[payoff_str]
        except KeyError as error:
            print(error)
            print("Please enter a valid payoff function. NOTE: payoff_func string is case sensitive")
            print("Make sure you exactly enter: cumulative ")