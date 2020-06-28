import copy
import networkx as nx
import statistics

from typing import List, Tuple, Dict
# import helper function to deprecate warnings
from helper_methods import deprecated
from src.gameconstruction import Graph

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

    def get_init_node(self) -> List[Tuple]:
        """
        A helper method to get the initial node(s) of a given graph stored in self.graph
        :return: node
        """
        # TODO: This methods loops through every element even after detecting a node which is a waste of time.
        #  Maybe I should break after I find an init node.
        # NOTE: if we assume to have only one init node then we can break immediately after we find it
        init_node = [node[0] for node in self.graph.nodes.data() if node[1].get('init') == True]

        return init_node

    def set_init_node(self, node) -> None:
        """
        A setter method to set a node as the init node
        :param node: a valid node of the graph
        """
        self.graph.nodes[node]['init'] = True

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

    def _convert_stack_to_play_str(self, stack: List[Tuple]) -> List[Tuple]:
        """
        Helper method to convert a play to its corresponding str representation
        :param stack: a dict of type {node: boolean_value}
        :return: basestring
        """
        # play = [node for node, value in stack.items() if value == True]
        # play_str = ''.join([ele for ele in stack])
        # lets create a list of tuple
        play_list: List[Tuple] = [node for node in stack]
        return play_list

    def _reinit_visitStack(self, stack: Dict[Tuple, bool]) -> Dict[Tuple, bool]:
        """
        helper method to re_initialize visit stack. For all nodes we re_initialize the value to be False
        :return: Dict of node with all the values as False Stack[node] = False
        """
        # RFE (request for enhancement): find better alternatives than traversing through the whole stack
        # IDEA: use generators or build methods from the builtin operator library in python
        # find all the values that are True and substitute False in there
        for node, flag in stack.items():
            if flag:
                stack[node] = False
        return stack

    def _compute_loop_value(self, stack: List) -> str:
        """
        A helper method to compute the value of a loop
        :param stack: a List of nodes (of type tuple)
        :return: The value associate with a play (nodes in stack) given a payoff function
        """

        def get_edge_weight(k: Tuple) -> float:
            return float(self.graph[k[0]][k[1]][0].get('weight'))

        # find the element which is repeated twice or more in the list
        # which has to be the very last element of the list
        assert stack.count(stack[-1]) >= 2, "The count of the repeated elements in a loops should be exactly 2"

        # get the index of the repeated node when it first appeared
        initial_index = stack.index(stack[-1])
        loop_edges = []
        # get the edge weights between the repeated nodes
        for i in range(initial_index, len(stack) - 1):
            # create the edge tuple up to the very last element
            loop_edges.append((stack[i], stack[i + 1]))

        return self.__payoff_func(map(get_edge_weight, [k for k in loop_edges]))

    def cycle_main(self) -> Dict[Tuple, str]:
        """
        A method to compute the all the possible loop values for a given graph with an init node. Before calling this
        function make sure you have already updated the init node and then call this function.
        :return: A dict consisting all the loops that exist in a graph with a given init node and its corresponding
        values for a given payoff function
        """
        visitStack: Dict[Tuple, bool] = {}
        # NOTE: the data player and the init flag cannot be accessed as graph[node]['init'/ 'player'] you have to first
        #  access the data as graph.nodes.data() and loop over the list each element in that list is a tuple
        #  (NOT A DICT) of the form (node_name, {key: value})

        # get all the info regarding the node of type tuple, format (node_name, {'player': value, 'init': True})
        init_node = [node[0] for node in self.graph.nodes.data() if node[1].get('init') == True]

        # initialize visitStack
        for node in self.graph.nodes:
            visitStack.update({node: False})

        # create a dict to hold values of the loops as str for a corresponding play which is a str too
        loop_dict: Dict[Tuple, str] = {}

        # visit each neighbour of the init node
        for node in self.graph.neighbors(init_node[0]):
            # reset all the flags except the init node flag to be False
            visitStack = self._reinit_visitStack(visitStack)
            visitStack[init_node[0]] = True
            nodeStack: List[Tuple] = []
            nodeStack.append(init_node[0])
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
                play_tuple = tuple(self._convert_stack_to_play_str(nodeStack))
                loop_value: str = self._compute_loop_value(nodeStack)
                loop_dict.update({play_tuple: loop_value})
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

    def compute_cVal(self, vertex: Tuple, debug:bool = False) -> str:
        """
        A Method to compute the cVal using  @vertex as the starting node
        :param vertex: a valid node of the graph
        :param debug: flag to print to print the max cooperative value from a given vertex
        :return: a single value of the max payoff when both adam and eve play cooperatively
        """
        # find all plays in which the vertex exist
        play_dict = {k: v for k, v in self.__loop_vals.items() if self._find_vertex_in_play(vertex, k)}

        # find the max of the value
        max_play = max(play_dict, key=lambda key: play_dict[key])
        if debug:
            print(f"The cVal from the node {vertex}")
            print(f"for the play {max_play} is {play_dict[max_play]}")

        return play_dict[max_play]

    def compute_aVal(self) -> None:
        raise NotImplementedError

    @deprecated
    def dump_mpg_data_file(self) -> None:
        """
        The format of the file should be <NODE> <MIN/MAX> <ADJ_NODE>:<EDGE_WEIGHT>...;
        :return:
        """

        # open the file and overwrite the context
        f = open("config/sample.mpg", "w")

        # helper function to print nodes
        def __print_nodes(_node):
            for n in _node:
                if 'v' in n:
                    f.write(f"{n[1:]}")
                else:
                    f.write(f"{n}")

        for node in self.graph.nodes():
            if self.graph.nodes[node]["player"] == "adam":
                player = "MIN"
            else:
                player = "MAX"

            # create the line

            # for curr_node in node:
            #     if 'v' in curr_node:
            #         f.write(f"{curr_node[1:]}")
            #     else:
            #         f.write(f"{curr_node} ")
            __print_nodes(node)
            f.write(f" {player} ")
            # create an edge list which will later be added to the line
            edge_list: List[Tuple] = []
            # get outgoing edges of a graph
            for e in self.graph.edges(node):
                edge_list.append((e[1], self.graph[e[0]][e[1]][0]['weight']))

            for ix, (k, v) in enumerate(edge_list):
                __print_nodes(k)
                if ix == len(edge_list) - 1:
                    f.write(f":{int(float(v))}; \n")
                else:
                    f.write(f":{int(float(v))}, ")

    def alt_dump_mpg_data_file(self) -> None:
        """
        The format of the file should be <NODE> <MIN/MAX> <ADJ_NODE>:<EDGE_WEIGHT>...;
        :return:
        """

        # open the file and overwrite the context
        f = open("config/sample.mpg", "w")

        for inode, node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]['map'] = inode

        for node in self.graph.nodes():
            if self.graph.nodes[node]["player"] == "adam":
                player = "MIN"
            else:
                player = "MAX"

            # create the line
            f.write(f"{self.graph.nodes[node]['map']} {player} ")
            # create an edge list which will later be added to the line
            edge_list: List[Tuple] = []
            # get outgoing edges of a graph
            for ie, e in enumerate(self.graph.edges(node)):
                # mapped_curr_node = self.graph.nodes[e[0]]['map']
                mapped_next_node = self.graph.nodes[e[1]]['map']
                # edge_list.append((, self.graph[e[0]][e[1]][0]['weight']))
                edge_weight = float(self.graph[e[0]][e[1]][0]['weight'])
                if ie == len(self.graph.edges(node)) - 1:
                    f.write(f"{mapped_next_node}:{int(edge_weight)}; \n")
                else:
                    f.write(f"{mapped_next_node}:{int(edge_weight)}, ")


if __name__ == "__main__":
    payoff_func = "mean"
    print(f"*****************Using {payoff_func}*****************")
    # construct graph
    G = Graph(False)
    org_graph = G.create_multigrpah()

    if payoff_func == "sup":
        gmax = G.construct_Gmax(org_graph)
        p = payoff_value(gmax, payoff_func)
    elif payoff_func == "inf":
        gmin = G.construct_Gmin(org_graph)
        p = payoff_value(gmin, payoff_func)
    else:
        p = payoff_value(org_graph, payoff_func)

    loop_vals = p.cycle_main()
    for k, v in loop_vals.items():
        print(f"Play: {k} : val: {v} ")

    p.alt_dump_mpg_data_file()

