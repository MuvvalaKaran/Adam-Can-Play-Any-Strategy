


class pyaoff_value():
    # a collection of different payoff function to construct the finite state machine

    # graph is the graph on which we will be computing all the payoff value
    def __init__(self, graph, payoff_func, vertices):
        self.graph = graph
        self.V = self.graph.nodes  # number of vertices
        self.payoff_func = payoff_func

    # write a basic code in which we detect cycles.
    def is_cyclic_util(self, v, visited, recStack):

        # Mark the current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # in recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.is_cyclic_util(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    def is_cyclic(self):
        # intialize visited and recStack as dict mapping each node to its boolean value
        # visited = [False] * len(self.V)
        # recStack = [False] * len(self.V)

        visited = {}
        recStack = {}
        for node in self.V:
            visited.update({node: False})
            recStack.update({node: False})
        for node in self.V:
            if visited[node] == False:
                if self.is_cyclic_util(node, visited, recStack) == True:
                    return True
        return False