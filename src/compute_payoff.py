import unittest
import copy
from helper_methods import deprecated
class payoff_value():
    # a collection of different payoff function to construct the finite state machine

    # graph is the graph on which we will be computing all the payoff value
    def __init__(self, graph, payoff_func, vertices):
        self.graph = graph
        self.V = self.graph.nodes  # number of vertices
        #  make sure that the string value of payoff match the key value exactly
        self.payoff_func = self.choose_payoff(payoff_func)

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
                loop_value = self.payoff_func(edgeStack)
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
        # visited = [False] * len(self.V)
        # recStack = [False] * len(self.V)
        # visited keeps track of nodes visited
        visited = {}
        # recStack keeps track of a loop
        recStack = {}
        loopStack = {}
        # edgeStack keeps track of the edge weights in the recStack
        edgeStack = []
        for node in self.V:
            visited.update({node: False})
            recStack.update({node: False})

        for node in self.V:
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

        for node, flag in stack.items():
            if flag:
                stack[node] = False
        # new_stack = {k: False for k, v in stack.items() - {k: False}}

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

        return self.payoff_func(map(get_edge_weight, [k for k in loop_edges]))

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
            # nodeStack.append()
            # edgeStack = self._reinit_visitStack(edgeStack)
            # edgeStack = []
            # edgeStack.append((init_node[0], node))
            # edgeStack[(init_node[0], node)] = True
            # edgeStack = []
            # edgeStack.append(self.graph[init_node[0]][node][0]['weight'])
            self.cycle_util(node, visitStack, loop_dict, edgeStack, nodeStack)

        return loop_dict

    def cycle_util(self, node, visitStack, loop_dict, edgeStack, nodeStack):
        # initialize loop flag as False and update the @vistStack with the current node as True
        visitStack = copy.copy(visitStack)
        visitStack[node] = True
        nodeStack = copy.copy((nodeStack))
        nodeStack.append(node)
        for neighbour in self.graph.neighbors(node):
            # edgeStack[(node, neighbour)] = True
            # check if the neighbour has been visited before
            # append edgeStack with the edge value between the current node = node and the next node = neighbour
            # edgeStack.append(self.graph[node][neighbour][0]['weight'])
            if visitStack[neighbour]:
                # edgeStack[(node, neighbour)] = True
                # self.cycle_util(neighbour)
                nodeStack = copy.copy((nodeStack))
                nodeStack.append(neighbour)
                # edgeStack.append((node, neighbour))
                play_str = self._conver_stack_to_play_str(nodeStack)
                loop_value = self._compute_loop_value(nodeStack)
                loop_dict.update({play_str: loop_value})
                nodeStack.pop()
                continue
                # edgeStack[(node, neighbour)] = False
                # return self.cycle_util(neighbour, visitStack, loop_dict, edgeStack, nodeStack)
            else:
                # nodeStack.append(neighbour)
                # edgeStack[(node, neighbour)] = True
                # edgeStack.append((node, neighbour))
                # nodeStack.append(neighbour)
                self.cycle_util(neighbour, visitStack, loop_dict, edgeStack, nodeStack)

