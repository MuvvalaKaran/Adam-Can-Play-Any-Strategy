# use this file to construct the game
# then upass this game to the payoff code to constrcut the finite state machine

from src.gameconstruction import Graph
from src.compute_payoff import payoff_value

def construct_graph():
    # testing imports
    G = Graph(True)
    # create the multigraph
    org_graph = G.create_multigrpah()
    gmin = G.construct_Gmin(org_graph)
    G.graph = gmin

    return G

def main():
    # construct graph
    graph = construct_graph()
    p = payoff_value(graph.graph, 'liminf', None)
    loop_vals = p.cycle_main()

    for k, v in loop_vals.items():
        print(f"Play: {k} : val: {v} ")
    # if p.is_cyclic() == True:
    #     print("There exists cycles")

if __name__ == "__main__":
    main()