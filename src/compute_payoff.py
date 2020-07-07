import copy
import networkx as nx
import statistics
import warnings
import sys

from collections import defaultdict
from typing import List, Tuple, Dict
from helper_methods import deprecated
from src.graph.graph import GraphFactory

"""
formatting notes : _protected variable: This effectively prevents it to be accessed unless, it is within a sub-class
                   __private variable : This gives a strong indication that this variable should not be touched from 
                   outside the class. Any attempt to do so will result in an AttributeError. 
"""
class payoff_value():
    # a collection of different payoff function to construct the finite state machine
    def __init__(self, graph: nx.MultiDiGraph, payoff_func: str) -> None:
        """
        A method to compute the value of an infinite loop on a given graph for a given payoff function.
        :param graph: Graph on which we would like to determine the values of a given play
        :param payoff_func: A function to efficiently quantify the value of a loop
        """
        self.graph = graph
        self.__V = self.graph.nodes
        self._raw_payoff_value: str = payoff_func
        self.__payoff_func = self._choose_payoff(self._raw_payoff_value)
        self.__loop_vals = None
        self._init_node = None
        self.initialize_init_node()

    def _choose_payoff(self, payoff_func: str):
        """
        A method to return the appropriate max/min function depending on the payoff function
        :param payoff_func: A function used to quantify the value of a loop
        :return: Callable - max or min
        """
        payoff_dict = {
            'limsup': max,
            'liminf': min,
            'inf': min,
            'sup': max,
            'mean': statistics.mean,
        }

        try:
            return payoff_dict[payoff_func]
        except KeyError as error:
            print(error)
            print("Please enter a valid payoff function. NOTE: payoff_func string is case sensitive")
            print("Make sure you exactly enter: 1.sup 2.inf 3. limsup 4. liminf 5. mean. ")

    def initialize_init_node(self) -> None:
        """
        A helper method to initialize the initial node
        :return:
        """
        for node in self.graph.nodes.data():
            if node[1].get('init'):
                self._init_node = node[0]
                return

    def get_init_node(self) -> Tuple:
        """
        A helper method to get the initial node(s) of a given graph stored in self.graph
        :return: node
        """
        # TODO: This methods loops through every element even after detecting a node which is a waste of time.
        #  Maybe I should break after I find an init node.
        # NOTE: if we assume to have only one init node then we can break immediately after we find it
        # init_node = [node[0] for node in self.graph.nodes.data() if node[1].get('init') == True]

        return self._init_node

    def set_init_node(self, node) -> None:
        """
        A setter method to set a node as the init node
        :param node: a valid node of the graph
        """
        self.graph.nodes[node]['init'] = True
        self._init_node = node

    def remove_attribute(self, tnode: Tuple, attr: str) -> None:
        """
        A method to remove a attribute @attr associated with a node @tnode. e.g weights, init are stored as dict keys
        and thus can be removed using the del operator or alternatively using this method
        :param tnode: the target node from which we would like to remove the corresponding attribute
        :param attr: a str/ name of the attribute that you would like to remove
        """
        try:
            self.graph.nodes[tnode].pop(attr)
        except KeyError as error:
            print(error)
            print(f"The node: {tnode} has no attribute : {attr}. Make sure the attribute is spelled correctly")

    def get_payoff_func(self) -> str:
        """
        a getter method to safely access the payoff function string
        :return: self.__payoff_func
        """
        return self._raw_payoff_value

    def _compute_loop_value(self, stack: List) -> str:
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
            loop_edge_w.append(float(self.graph[stack[i]][stack[i + 1]][0].get('weight')))

        return self.__payoff_func(loop_edge_w)

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
        for node in self.graph.neighbors(self._init_node):
            visitStack: Dict[Tuple, bool] = defaultdict(lambda: False)
            visitStack[self._init_node] = True
            nodeStack: List[Tuple] = []
            nodeStack.append(self._init_node)
            self._cycle_util(node, visitStack, loop_dict, nodeStack)

        self.__loop_vals = loop_dict
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
        for neighbour in self.graph.neighbors(node):
            if visitStack[neighbour]:
                nodeStack = copy.copy((nodeStack))
                nodeStack.append(neighbour)
                loop_value: str = self._compute_loop_value(nodeStack)
                loop_dict.update({tuple(nodeStack): loop_value})
                nodeStack.pop()
                continue
            else:
                self._cycle_util(neighbour, visitStack, loop_dict, nodeStack)

    def _find_vertex_in_play(self, v: Tuple, play: Tuple) -> bool:
        """
        A helper method to check if a node exists in a corresponding play or not
        :param v: Node to search for in the given play @play
        :param play: a sequence of nodes on the given graph
        :return: True if the vertex exist in the @play else False
        """
        if v in play:
            return True
        return False

    def compute_cVal(self, vertex: Tuple, debug: bool = False) -> str:
        """
        A Method to compute the cVal using  @vertex as the starting node
        :param vertex: a valid node of the graph
        :param debug: flag to print to print the max cooperative value from a given vertex
        :return: a single value of the max payoff when both adam and eve play cooperatively
        """
        # find all plays in which the vertex exist
        play_dict = {k: v for k, v in self.__loop_vals.items() if self._find_vertex_in_play(vertex, k)}

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

        return play_dict[max_play]

if __name__ == "__main__":
    payoff_func = "inf"
    print(f"*****************Using {payoff_func}*****************")

    if payoff_func == "sup":
        _graph = GraphFactory._construct_gmin_graph(debug=False, plot=True)
        p = payoff_value(_graph._graph, payoff_func)
    elif payoff_func == "inf":
        _graph = GraphFactory._construct_gmax_graph(debug=False, plot=True)
        p = payoff_value(_graph._graph, payoff_func)
    else:
        _graph = GraphFactory._construct_product_automaton_graph(use_alias=False, scLTL_formula="!b & Fc",
                                                                 plot=True, prune=False)
        p = payoff_value(_graph._graph, payoff_func)

    loop_vals = p.cycle_main()
    for k, v in loop_vals.items():
        print(f"Play: {k} : val: {v} ")

