import copy
import warnings
import sys
import statistics

from collections import defaultdict
from typing import Tuple, List, Dict

# import local packages
from .base import Payoff
from src.factory.builder import Builder
from src.graph.base import Graph


class InfinitePayoff(Payoff):
    """
    This is the abstract base class that describe a payoff function and its functionality

    :param: graph:          Graph on which we identify all the possible loops
                            quantify the value based on the payoff function being used
    :param: __payoff_func:  a string specifying the type of payoff function we will be implementing
                            e.g liminf, limsup etc.
    """

    def __init__(self, graph: Graph, payoff: callable) -> 'Payoff()':
        # self.graph = graph
        # self.__V = self.graph._graph.nodes
        # self.payoff_func: callable = payoff
        # self.__loop_vals = None
        # self.init_node = None
        # self.initialize_init_node()
        Payoff.__init__(self, graph=graph, payoff=payoff)

    def _compute_loop_value(self, stack: List) -> float:
        """
        A helper method to compute the value of a loop
        :param stack: a List of nodes (of type tuple)
        :return: The value associate with a play (nodes in stack) given a payoff function
        """
        # get the index of the repeated node when it first appeared
        initial_index = stack.index(stack[-1])
        loop_edge_w = []
        # get the edge weights between the repeated nodes
        for i in range(initial_index, len(stack) - 1):
            # create the edge tuple up to the very last element
            loop_edge_w.append(float(self.graph._graph[stack[i]][stack[i + 1]][0].get('weight')))

        return self.payoff_func(loop_edge_w)

    def cycle_main(self) -> Dict[Tuple, str]:
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
        loop_dict: Dict[Tuple, str] = {}

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
                loop_value: str = self._compute_loop_value(nodeStack)
                loop_dict.update({tuple(nodeStack): loop_value})
                continue

            self._cycle_util(node, visitStack, loop_dict, nodeStack)

        self._loop_vals = loop_dict
        return loop_dict

    def _cycle_util(self, node, visitStack: Dict[Tuple, bool], loop_dict: Dict[Tuple, str],
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

                loop_value: str = self._compute_loop_value(nodeStack)
                loop_dict.update({tuple(nodeStack): loop_value})
                nodeStack.pop()
                continue
            else:
                self._cycle_util(neighbour, visitStack, loop_dict, nodeStack)


class InfinitePayoffBuilder(Builder):

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

        self._instance = InfinitePayoff(graph=graph,
                                payoff=payoff_val)

        return self._instance

    def _get_payoff_val(self, payoff_str: str):

        payoff_dict = {
            'limsup': max,
            'liminf': min,
            'inf': min,
            'sup': max,
            'mean': statistics.mean,
        }

        try:
            return payoff_dict[payoff_str]
        except KeyError as error:
            print(error)
            print("Please enter a valid payoff function. NOTE: payoff_func string is case sensitive")
            print("Make sure you exactly enter: 1.sup 2.inf 3. limsup 4. liminf 5. mean. ")