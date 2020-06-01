import unittest
import copy
import re
import operator

# import helper function to depreceate warnings
from helper_methods import deprecated

"""
formatting notes : _protected variable: This effectively prevents it to be accessed unless, it is within a sub-class
                   __private variable : This gives a strong indication that this variable should not be touched from 
                   outside the class. Any attempt to do so will result in an AttributeError. 
"""
class payoff_value():
    # a collection of different payoff function to construct the finite state machine

    # graph is the graph on which we will be computing all the payoff value
    def __init__(self, graph, payoff_func, vertices):
        self.graph = graph
        self.__V = self.graph.nodes  # number of vertices
        #  make sure that the string value of payoff match the key value exactly
        self.__payoff_func = self.choose_payoff(payoff_func)
        self.__loop_vals = None

    def choose_payoff(self, payoff_func):
        payoff_dict = {
            'limsup': max,
            'liminf': min,
            'inf': min,
            'sup': max
        }

        return payoff_dict[payoff_func]

    @deprecated
    # write a basic code in which we detect cycles
    def is_cyclic_util(self, v, visited, recStack, edgeStack):

        # Mark the current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True
        # edgeStack = []
        # Recur for all neighbours
        # if any neighbour is visited and in
        # in recStack then graph is cyclic
        for neighbour in self.graph[v]:
            # update the edgeStack
            edgeStack.append(self.graph[v][neighbour][0]['weight'])

            # if the neighbour hasn't been visited then mark that as visited and then look at its
            # neighbours
            if not visited[neighbour]:
                recStack, loop_value = self.is_cyclic_util(neighbour, visited, recStack, edgeStack)
                return recStack, loop_value
                    # the sequence of nodes
                    # play = [node for node, value in recStack.items() if value == True]
                    # # convert it into a str so that it can stored are a dect key
                    # play_str = ''.join([str(ele) for ele in play])
                    # loop_value = self.payoff_func(edgeStack)
                    # edgeStack.clear()
                    # return recStack, loop_value
                    # return True

            # if cycle exists then return True
            elif recStack[neighbour]:
                # compute the min/max
                loop_value = self.__payoff_func(edgeStack)
                edgeStack.clear()
                return recStack, loop_value
                # return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    @deprecated
    def is_cyclic(self):
        # initialize visited and recStack as dict mapping each node to its boolean value
        # visited = [False] * len(self.__V)
        # recStack = [False] * len(self.__V)
        # visited keeps track of nodes visited
        visited = {}
        # recStack keeps track of a loop
        recStack = {}
        loopStack = {}
        # edgeStack keeps track of the edge weights in the recStack
        edgeStack = []
        for node in self.__V:
            visited.update({node: False})
            recStack.update({node: False})

        for node in self.__V:
            if not visited[node]:
                play, weight = self.is_cyclic_util(node, visited, recStack, edgeStack)
                play = [node for node, value in recStack.items() if value == True]
                # convert it into a str so that it can be stored as a dict key
                play_str = ''.join([str(ele) for ele in play])
                # update loopStack
                loopStack.update({play_str: weight})
        return False

    def _conver_stack_to_play_str(self, stack):
        """
        Helper method to convert a play to its corresponding str representation
        :param stack: a dict of type {node: boolean_value}
        :return: basestring
        """
        # play = [node for node, value in stack.items() if value == True]
        play_str = ''.join([str(ele) for ele in stack])
        return play_str

    def _reinit_visitStack(self, stack):
        """
        helper method to re_initalize visit stack
        :return:
        """
        # RFE (request for enhancement): find better alternative than traversing through the whole loop
        # IDEA: use generators or build methods from the operator library
        # find all the values that are True and substitute False in there
        for node, flag in stack.items():
            if flag:
                stack[node] = False
        return stack

    def _compute_loop_value(self, stack):
        """
        helper method to compute the value of a loop
        :param stack:
        :return:
        """

        def get_edge_weight(k):
            # k is a tuple of format (curr_node, adj_node)
            # getter method for edge weights
            return self.graph[k[0]][k[1]][0]['weight']
        # loop_edges = []
        #
        # visited_edges = dict(filter(lambda elem: elem[1] == True, stack.items()))
        # for k in visited_edges.keys():
        #     # if v == True:
        #     loop_edges.append(get_edge_weight(k))
        # [1,3,3]
        # find the element which is repeated twice in the list - which has to be the very last element of the list
        assert stack.count(stack[-1]) == 2, "The count of the repeated elements in a loops should be exactly 2"

        # get the index of the repeated node when it first appeared
        initial_index = stack.index(stack[-1])
        loop_edges = []
        # get the edge weights between the repeated nodes
        for i in range(initial_index, len(stack) - 1):
            # create the edge tuple upto the very last element
            loop_edges.append((stack[i], stack[i + 1]))

        return self.__payoff_func(map(get_edge_weight, [k for k in loop_edges]))

    def cycle_main(self):
        visitStack = {}
        edgeStack = {}
        """the data player and the init flag cannot be accessed as graph[node]['init'/ 'player']
            you have to first access the data as graph.nodes.data() and loop over the list
            each element in that list is a tuple (NOT A DICT) of the form (node_name, {key: value})"""

        # get all the info regarding the node of type tuple, format (node_name, {'player': value, 'init': True})
        init_node = [node[0] for node in self.graph.nodes.data() if node[1].get('init') == True]

        # initialize visitStack
        for node in self.graph.nodes:
            visitStack.update({node: False})

        # initialize edgeStack
        for edge in self.graph.edges():
            # the keys are stored in the format of a tuple (curr_node, adj_node)
            edgeStack.update({edge: False})

        # create a dict to hold values of the loops
        loop_dict = {}

        # visit each neighbour of the init node
        for node in self.graph.neighbors(init_node[0]):
            # reset all the flags except the init node flag to be False
            visitStack = self._reinit_visitStack(visitStack)
            visitStack[init_node[0]] = True
            nodeStack = []
            nodeStack.append(init_node[0])
            self.cycle_util(node, visitStack, loop_dict, edgeStack, nodeStack)

        self.__loop_vals = loop_dict
        return loop_dict

    def cycle_util(self, node, visitStack, loop_dict, edgeStack, nodeStack):
        # initialize loop flag as False and update the @visitStack with the current node as True
        visitStack = copy.copy(visitStack)
        visitStack[node] = True
        nodeStack = copy.copy((nodeStack))
        nodeStack.append(node)
        for neighbour in self.graph.neighbors(node):
            if visitStack[neighbour]:
                nodeStack = copy.copy((nodeStack))
                nodeStack.append(neighbour)
                play_str = self._conver_stack_to_play_str(nodeStack)
                loop_value = self._compute_loop_value(nodeStack)
                loop_dict.update({play_str: loop_value})
                nodeStack.pop()
                continue
            else:
                self.cycle_util(neighbour, visitStack, loop_dict, edgeStack, nodeStack)

    def _find_vertex_in_play(self, v, play):
        """
        A helper method to to check is a node is in a player or not
        :param v: name of the node to search for
        :type @node of type networkx MG
        :param play: a sequence of nodes/play on the given graph
        :type basestring
        :return: True if the vertex exist in the play else False
        :type bool
        """

        # if there is a match then return True
        # took the long route rather than just re.search({v}, txt) as I could not enter v directly even with
        # escape sequence characters
        node_re = re.compile(f'{v}')
        if re.search(node_re.pattern, play):
            return True
        return False

    def compute_cVal(self, vertex):
        """
        Method to compute the cVal using  @vertex as the starting node
        :param vertex: a valid node of the graph
        :type: @node
        :return: a single value of the max payoff when both adam and eve play cooperatively
        :type: int/float
        """
        # compute the Val for various loops that exist in the graph and then choose the play with the max Val

        # find all plays in which the vertex exist
        play_dict = {k: v for k, v in self.__loop_vals.items() if self._find_vertex_in_play(vertex, k)}

        # find the max of the value
        max_play = max(play_dict, key=lambda key: play_dict[key])
        print(f"The cVal from the node {vertex}") # is {list(play_dict.values())[max_pos_index]} "
        print(f"for the play {max_play} is {play_dict[max_play]}")

        return play_dict[max_play]


    def compute_aVal(self):
        raise NotImplementedError

