import copy
import warnings
import sys
import statistics
import abc
# import hash

from bidict import bidict
from abc import abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

# import local packages
from src.factory.builder import Builder
from src.graph.base import Graph


class Payoff(abc.ABC):
    """
    This is the abstract base class that describe a payoff function and its functionality

    :param: graph:          Graph on which we identify all the possible loops
                            quantify the value based on the payoff function being used
    :param: __payoff_func:  a string specifying the type of payoff function we will be implementing
                            e.g liminf, limsup etc.
    """

    def __init__(self, graph: Graph, payoff: callable) -> 'Payoff()':
        self.graph = graph
        self.map_tuple_idx = bidict({})
        self.__V = self.graph._graph.nodes
        self.payoff_func: callable = payoff
        self._loop_vals: Optional[Dict[tuple, float]] = None
        self.init_node = None
        self.initialize_init_node()

    def initialize_init_node(self) -> None:
        """
        A helper method to initialize the initial node
        :return:
        """
        for node in self.graph._graph.nodes.data():
            if node[1].get('init'):
                self.set_init_node(node[0])
                break

    # @property
    def get_init_node(self) -> Tuple:
        """
        A helper method to get the initial node(s) of a given graph stored in self.graph
        :return: node
        """

        return self.init_node

    # @property
    def get_payoff_func(self) -> callable:
        """
        A getter for payoff_func attr which is callable that stores the payoff callable
        :return: callable like min, max, mean, and sum
        """

        return self.payoff_func

    @property
    def loop_vals(self):
        return self._loop_vals

    # @init_node.setter
    def set_init_node(self, node):
        """
        A setter method to set a node as the init node
        :param node: a valid node of the graph
        """
        try:
            self.graph._graph.nodes[node]['init'] = True
        except KeyError:
            raise KeyError(f"Please ensure that {node} is a valid node of Graph {self.graph._graph_name}")

        self.init_node = node

    def create_tuple_mapping(self, play: tuple):
        """
        A bidirectional dictionary hashes a tuple and stores its hash value as the value
        :return:
        """
        self.map_tuple_idx.update({play: hash(play)})

    def get_map_tuple_idx(self) -> bidict:
        return self.map_tuple_idx

    def remove_attribute(self, tnode: Tuple, attr: str) -> None:
        """
        A method to remove a attribute @attr associated with a node @tnode. e.g weights, init are stored as dict keys
        and thus can be removed using the del operator or alternatively using this method
        :param tnode: the target node from which we would like to remove the corresponding attribute
        :param attr: a str/ name of the attribute that you would like to remove
        """
        try:
            self.graph._graph.nodes[tnode].pop(attr)
        except KeyError as error:
            print(error)
            print(f"The node: {tnode} has no attribute : {attr}. Make sure the attribute is spelled correctly")

    @abstractmethod
    def _compute_loop_value(self, stack: List) -> str:
        pass

    @abstractmethod
    def cycle_main(self) -> Dict[Tuple, str]:
        pass

    @abstractmethod
    def _cycle_util(self, node, visitStack: Dict[Tuple, bool], loop_dict: Dict[Tuple, str],
                    nodeStack: List[Tuple]) -> None:
        pass

    def _find_vertex_in_play(self, v: Tuple, play: Tuple) -> bool:
        """
        A helper method to check if a node exists in a corresponding play or not
        :param v: Node to search for in the given play @play
        :param play: a sequence of nodes on the given graph
        :return: True if the vertex exist in the @play else False
        """
        if v in self.map_tuple_idx.inverse[play]:
            return True
        return False

    def compute_cVal(self, vertex: Tuple, debug: bool = False) -> float:
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

        return play_dict[max_play]