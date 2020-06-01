# use this file to construct the game
# then upass this game to the payoff code to constrcut the finite state machine
import math
import copy
import networkx as nx
import sys

from src.gameconstruction import Graph
from src.compute_payoff import payoff_value

# assert that this code runs on linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"


def construct_graph():
    # testing imports
    G = Graph(True)
    # create the multigraph
    # FIXME: add code that automatically compute gmin/ gmax based on the payoff function pased by the user
    org_graph = G.create_multigrpah()
    gmin = G.construct_Gmin(org_graph)
    G.graph = org_graph

    return G


def construct_alt_game(graph, edge):
    # remove the edge of the form (u, v) from the graph
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(edge[0], edge[1])
    return new_graph

def compute_w_prime(payoff_handle, graph):
    # compute W prime
    # calculate the loop values
    loop_vals = payoff_handle.cycle_main()
    for k, v in loop_vals.items():
        print(f"Play: {k} : val: {v} ")

    # # compute the cVal/aVal
    # cval = payoff_handle.compute_cVal(1)
    w_prime = {}
    for edge in graph.graph.edges():
        if graph.graph.nodes(data='player')[edge[0]] == 'adam':
            # w_prime.add(-1 * math.inf)
            w_prime.update({edge: -1 * math.inf})
        else:

            # get a list of all the cVal for all the alternate edges
            tmp_graph = construct_alt_game(graph.graph, edge)
            # costruct the game without the org edge and find the max from each alternate play
            tmp_cvals = []
            payoff_handle.graph = tmp_graph
            payoff_handle.cycle_main()

            for e in tmp_graph.out_edges(edge[0]):
                # get all the edges from the give node
                tmp_cvals.append(payoff_handle.compute_cVal(e[1]))
            if len(tmp_cvals) != 0:
                w_prime.update({edge: max(tmp_cvals)})
            else:
                # TODO: check if this correct or not?!
                payoff_handle.graph = graph.graph
                payoff_handle.cycle_main()
                w_prime.update({edge: payoff_handle.compute_cVal(edge[1])})
                # w_prime.add(max(tmp_cvals))
    return w_prime

def construct_g_b(g_hat, org_graph, b, w_prime):
    # G_b = nx.MultiDiGraph(name=f"G_{b}")
    g_hat.add_nodes_from([f"{n}_{b}" for n in org_graph.nodes()])

    # assign each node a player if it has'nt been initialized yet
    for n in g_hat.nodes():
        # assign the nodes of G_b with 1 in it at n[0] to have a 'init' attribute
        if len(n) > 1 and n[0] == '1':
            g_hat.nodes[n]['init'] = True
        if g_hat.nodes(data='player')[n] is None:
            if org_graph.nodes(data='player')[int(n[0])] == "adam":
                g_hat.nodes[n]['player'] = "adam"
            else:
                g_hat.nodes[n]['player'] = "eve"

    for e in org_graph.edges():
        if w_prime[e] <= b:
            g_hat.add_edge(f"{e[0]}_{b}", f"{e[1]}_{b}")
    # TODO check is if it efficient to add w_hat here or afterwards by looping over all the edges in G_hat
    # return graph


def get_max_weight(graph):
    """
    A helper method to compute the max weight in a given graph
    :param graph:
    :return:
    """

    # max(play_dict, key=lambda key: play_dict[key])
    # max(graph.edges(data='weight'), key= lambda key:graph.edges(data='weight')[key])
    weight_list = []
    for _, _, weight in graph.edges(data='weight'):
        weight_list.append(weight)

    return max(weight_list)


def construct_g_hat(org_graph, w_prime):
    # construct new graph according to the pseudocode 3
    G_hat = nx.MultiDiGraph(name="G_hat")
    G_hat.add_nodes_from(['0', '1', 'T'])
    G_hat.nodes['0']['player'] = "adam"
    G_hat.nodes['1']['player'] = "eve"
    G_hat.nodes['T']['player'] = "eve"
    # add the edges with the weights
    G_hat.add_weighted_edges_from([('0', '0', 0), ('T', 'T', -2 * get_max_weight(org_graph) - 1)])

    # compute the range of w_prime function
    w_set = set(w_prime.values()) - {-1 * math.inf}
    # construct g_b
    for b in w_set:
        construct_g_b(G_hat, org_graph, b, w_prime)

    # add edges between 1 of G_hat and init(1_b) of graph G_b with edge weights 0
    for b in w_set:
        G_hat.add_weighted_edges_from([('1', f"1_{b}", 0)])

    # TODO: replace @w_prime(e) = -inf for e that belong to adam substitution to the org edge value in the org_graph
    def w_hat_b(_org_graph, org_edge, b_value):
        if w_prime[org_edge] != -1 * math.inf:
            return w_prime[org_edge] - b_value
        else:
            try:
                return _org_graph[org_edge[0]][org_edge[1]][0].get('weight') - b_value
            except KeyError:
                print(KeyError)
                print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                      "original graph has been modified and the edge {} does not exist".format(org_edge))

    # add edges with their respective weights
    for e in G_hat.edges():
        # only add weights if has'nt been initialized
        if G_hat[e[0]][e[1]][0].get('weight') is None:
            # initialize_weights
            # the nodes are stored as string in format "1_1" so we need only the first element
            # condition to check if the node belongs to g_b or not
            if len(e[0]) > 1:
                G_hat[e[0]][e[1]][0]['weight'] = w_hat_b(org_graph, (int(e[0][0]),
                                                         int(e[1][0])),
                                                         int(e[0][-1]))

def main():

    # construct graph
    graph = construct_graph()
    p = payoff_value(graph.graph, 'limsup', None)

    # construct W prime
    w_prime = compute_w_prime(p, graph)

    # construct G_hat
    construct_g_hat(graph.graph, w_prime)
    # Compute antaganostic value

if __name__ == "__main__":
    main()