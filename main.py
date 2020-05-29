# use this file to construct the game
# then upass this game to the payoff code to constrcut the finite state machine

from src.gameconstruction import Graph
from src.compute_payoff import pyaoff_value

def construct_graph():
    # testing imports
    G = Graph(True)
    # create the multigraph
    org_graph = G.create_multigrpah()
    G.graph = org_graph

    return G

def main():
    # construct graph
    graph = construct_graph()
    p = pyaoff_value(graph.graph, None, None)
    if p.is_cyclic() == True:
        print("There exists cycles")

if __name__ == "__main__":
    main()