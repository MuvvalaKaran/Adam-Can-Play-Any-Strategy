# use this file to construct the game
# then pass this game to the payoff code to construct the finite state machine
import math
import copy
import networkx as nx
import sys
import re
import warnings
import operator

from typing import List, Tuple
from src.gameconstruction import Graph
from src.compute_payoff import payoff_value
from helper_methods import deprecated

# asserts that this code is tested in linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"


def construct_graph(payoff_func: str, *args, **kwargs) -> Graph:
    G: Graph = Graph(False)
    # create the directed multi-graph
    org_graph = G.create_multigrpah()

    sup_re = re.compile('^sup$')
    inf_re = re.compile('^inf$')

    if inf_re.match(payoff_func):
        gmin = G.construct_Gmin(org_graph)
        G.graph = gmin
    elif sup_re.match(payoff_func):
        gmax = G.construct_Gmax(org_graph)
        G.graph = gmax
    else:
        G.graph = org_graph
    return G


def construct_alt_game(graph, edge):
    # remove the edge of the form (u, v) from the graph
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(edge[0], edge[1])
    return new_graph


def compute_w_prime(payoff_handle, graph):
    print("*****************Constructing W_prime*****************")
    # compute W prime
    # calculate the loop values
    _loop_vals = payoff_handle.cycle_main()
    for k, v in _loop_vals.items():
        print(f"Play: {k} : val: {v} ")

    # compute the cVal/aVal
    w_prime = {}
    for edge in graph.graph.edges():
        if graph.graph.nodes(data='player')[edge[0]] == 'adam':
            # w_prime.add(-1 * math.inf)
            w_prime.update({edge: -1 * math.inf})
        else:
            # get a list of all the cVal for all the alternate edges
            # TODO: check if this is necessary as @tmp_graph is already a copy and I am making a deepcopy of a deepcopy.
            #  Seems redundant but safety over everything.
            tmp_graph = construct_alt_game(graph.graph, edge)
            # construct the game without the org edge and find the max from each alternate play
            tmp_cvals = []
            payoff_handle.graph = tmp_graph
            payoff_handle.cycle_main()

            for e in tmp_graph.out_edges(edge[0]):
                # just to be safe, we will work with a copy
                tmp_copied_graph = copy.deepcopy(tmp_graph)
                # construct a new graph with e[1] as the initial vertex and compute loop_vals again and then
                # proceed ahead in this method
                # 1. remove the current init node of the graph
                # 2. add e[1] as the new init vertex
                # 3. compute the loop vals for this new graph
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                # tmp_payoff_handle.graph = tmp_copied_graph
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(e[1])
                tmp_payoff_handle.cycle_main()

                # get all the edges from the give node
                tmp_cvals.append(tmp_payoff_handle.compute_cVal(e[1]))
            if len(tmp_cvals) != 0:
                w_prime.update({edge: max(tmp_cvals)})
            else:
                # TODO: check if this correct or not?!
                # make a copy of the org_graph and a another tmp_payoff_handle and compute loop vals
                # payoff_handle.graph = graph.graph
                tmp_copied_graph = copy.deepcopy(graph.graph)
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(edge[1])
                tmp_payoff_handle.cycle_main()
                w_prime({edge: tmp_payoff_handle.compute_cVal(edge[1])})
                # payoff_handle.cycle_main()
                # w_prime.update({edge: payoff_handle.compute_cVal(edge[1])})
                # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())

    print(f"the value of b are {set(w_prime.values())}")

    return w_prime, _loop_vals


def _construct_g_b(g_hat, org_graph, b, w_prime):
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
    print("*****************Constructing G_hat*****************")
    # construct new graph according to the pseudocode 3
    G_hat = nx.MultiDiGraph(name="G_hat")
    G_hat.add_nodes_from(['0', '1', 'T'])
    G_hat.nodes['0']['player'] = "adam"
    G_hat.nodes['1']['player'] = "eve"
    G_hat.nodes['T']['player'] = "eve"

    # add v0 as the initial node
    G_hat.nodes['0']['init'] = True
    # add the edges with the weights
    G_hat.add_weighted_edges_from([('0', '0', 0), ('0', '1', 0), ('T', 'T', -2 * get_max_weight(org_graph) - 1)])

    # compute the range of w_prime function
    w_set = set(w_prime.values()) - {-1 * math.inf}
    # construct g_b
    for b in w_set:
        _construct_g_b(G_hat, org_graph, b, w_prime)

    # add edges between 1 of G_hat and init(1_b) of graph G_b with edge weights 0
    for b in w_set:
        G_hat.add_weighted_edges_from([('1', f"1_{b}", 0)])

    def remove_attribute(G, tnode, attr):
        G.nodes[tnode].pop(attr, None)

    # TODO: looks like a tmp patch; this may or may not work when using sup/inf payoff function.
    #  Need to verify it
    # remove nodes from g_b which has init nodes
    _init_node = [node[0] for node in G_hat.nodes.data() if node[1].get('init') == True]
    g_b_node_pattern = re.compile('_')
    for duplicate_init_node in _init_node:
        if g_b_node_pattern.search(duplicate_init_node) is not None:
            remove_attribute(G_hat, duplicate_init_node, 'init')

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

    # for nodes that don't have any outgoing edges add a transition to the terminal node i.e 'T' in our case
    for node in G_hat.nodes():
        if G_hat.out_degree(node) == 0:
            # add transition to the terminal node
            G_hat.add_weighted_edges_from([(node, 'T', 0)])

    return G_hat


def plot_graph(graph, file_name, save_flag: bool = True):
    print(f"*****************Plotting graph with save_flag = {save_flag}*****************")
    # create Graph object
    plot_handle = Graph(save_flag)
    plot_handle.graph = graph

    # file to store the yaml for plotting it in graphviz
    plot_handle.file_name = file_name

    # dump the graph to yaml
    plot_handle.dump_to_yaml()

    # plot graph
    plot_handle.graph_yaml = plot_handle.read_yaml_file(plot_handle.file_name)
    plot_handle.plot_fancy_graph()


def compute_aVal(g_hat, meta_b, _Val_func, w_prime, _loop_vals):
    """
    A function to compute the regret value according to algorithm 4 : Reg = -1 * Val(.,.) on g_hat
    :param g_hat: a directed multi-graph constructed using construct_g_hat()
    :type @Digraph
    :param meta_b: a real valued number that belong to set(w_prime)
    :type float
    :param Val: a payoff function that belong to {limsup, liminf, sup, inf}
    :type basestring
    :return: a dict consisting of the reg value and strategy for eve and adam respectively
    :type dict
    """
    print(f"*****************Computing regret and strategies for eve and adam*****************")
    assert (meta_b in set(w_prime.values()) - {-1*math.inf}), "make sure that the b value manually entered belongs " \
                                                              "to w_prime i.e {}.".format(w_prime)

    # create a dict which will be update once we find the reg value and strategies for eve and adam
    str_dict = {
        'reg': None,
        'eve': None,
        'adam': None
    }

    # create a empty dict to hold the strategy for eve and adam
    eve_str = {}
    adam_str = {}

    # update 0 to 1 transition in g_hat and 1 to 1_b depending on the value of b - hyper-paramter chosen by the user
    # NOTE: here the strategy is a dict; key is the current node while the value is the next node.
    #  This would NOT work in condition where there are more than one edge between the same nodes i.e 0-1 with multiple
    #  weights. If that is the case then we need to change the implementation to store the whole edge instead of just
    #  the next node.
    adam_str.update({'0': "1"})
    _max_accepted_b = []
    for b in set(w_prime.values()) - {-1*math.inf}:
        if b <= meta_b:
            # add all those transition that satisfy the condition
            _max_accepted_b.append(b)

    # get the max_b below the accepted val (meta_b)
    allowed_b = max(_max_accepted_b)
    eve_str.update({"1": f"1_{allowed_b}"})

    for node in g_hat.nodes():
        # I manually add the initial transitions of node 0 and 1 of g_hat graph
        # TODO: change the second condition to <= if you want to compute all the str for all nodes <= @allowed_b
        if len(node) > 1 and int(node[-1]) == allowed_b:
            # if the node belongs to adam
            if g_hat.nodes[node]['player'] == 'adam':
                # get the next node and update
                adam_str.update({node: _get_next_node(g_hat, node, min)})
            # if node belongs to eve
            elif g_hat.nodes[node]['player'] == 'eve':
                eve_str.update({node: _get_next_node(g_hat, node, max)})
            else:
                raise warnings.warn(f"The node {node} does not belong either to eve or adam. This should have "
                                    f"never happened")

    # now given the strategy compute the regret using the Val function
    # 1. find a loop
    # 2. pass it to _Val function which is the value of that loop for the corresponding value function
    # 3. update the str_dict['reg'] value
    # merging both the dictionaries
    a_val = _play_loop(g_hat, {**eve_str, **adam_str}, _Val_func)
    # a_val = _Val(_loop_vals, loop_str)
    str_dict['reg'] = -1*a_val

    # update eve and adam str and return it
    str_dict['adam'] = adam_str
    str_dict['eve'] = eve_str

    return str_dict


def _get_init_node(graph):
    """
    A helper method to get the initial node of a given graph
    :param graph:
    :return:
    """
    init_node = [node[0] for node in graph.nodes.data() if node[1].get('init') == True]

    return init_node[0]


def _play_loop(graph, strategy, payoff_func):
    """
    helper method to find a loop while following str on g_hat and return a corresponding str sequence of nodes
    :param graph:
    :type @ Digraph
    :param strategy:
    :type dict
    :return: sequence of nodes e.g '0121'
    :type basestring
    """
    # add nodes to this stack and as soon as a loop is found we break
    play = [_get_init_node(graph)]
    # weigth = []
    # for node in graph.nodes():
    while 1:
        # NOTE: This is assuming that a strategy is deterministic i.e the next node is only 1
        # play.append(node)
        play.append(strategy[play[-1]])
        # weigth.append(graph[play[-1]][strategy[play[-1]]][0]['weight'])

        if play.count(play[-1]) == 2:
            play_str = ''.join([str(ele) for ele in play])
            # pop the very last element as it is repeated twice
            # play.pop()

            # create a tmp graph with the current node with their respective edges, compute the val and return it
            str_graph = nx.MultiDiGraph(name="str_graph")
            str_graph.add_nodes_from(play)
            for i in range(0, len(play) - 1):
                str_graph.add_weighted_edges_from([(play[i],
                                                    play[i+1],
                                                    graph[play[i]][play[i+1]][0].get('weight'))
                                                   ]
                                                  )

            # manually add an edge from last to last -1 to complete the cycle
            str_graph.add_weighted_edges_from([(play[-1],
                                                play[-2],
                                                graph[play[-1]][play[-2]][0].get('weight'))])

            # add init node
            str_graph.nodes[play[0]]['init'] = True
            # add this graph to compute_payoff class
            tmp_p_handle = payoff_value(str_graph, payoff_func)
            _loop_vals = tmp_p_handle.cycle_main()

            return _loop_vals[play_str]


@deprecated
def _Val(loop_dict, play):
    # just a helper function mapping a play to its real value.
    try:
        return loop_dict[play]
    except KeyError as error:
        print(error)
        print(f"The play {play} does not exist. This might be due to the fact that this is a not a val loop or or "
              f"the play for some reason is missing from the computation in w_prime() method")

def _get_next_node(graph, curr_node, func):
    # return the next node for eve and adam on g_hat game
    # func - either max or min
    assert (func == max or func == min), "Please make sure the deciding function for transitions on the game g_hat for " \
                                         "eve and adam is either max or min"

    wt_list = {}
    for adj_edge in graph.edges(curr_node):
        # get the edge weight, store it in a list and find the max/min and return the next_node
        # G_hat[e[0]][e[1]][0].get('weight')
        wt_list.update({adj_edge : graph[adj_edge[0]][adj_edge[1]][0].get('weight')} )
        # wt_list.append(graph[adj_edge[0]][adj_edge[1]][0].get('weight'))

    next_node = func(wt_list.items(), key=operator.itemgetter(1))[0]

    return next_node[1]


def main():
    payoff_func = "inf"
    print(f"*****************Using {payoff_func}*****************")
    # construct graph
    graph = construct_graph(payoff_func)
    p = payoff_value(graph.graph, payoff_func)

    # FIXME: fails when using inf/sup payoff function
    # construct W prime
    w_prime, loop_vals = compute_w_prime(p, graph)

    # construct G_hat
    G_hat = construct_g_hat(graph.graph, w_prime)

    # use methods from the Graph class create a visualization
    plot_graph(G_hat, file_name='src/config/g_hat_graph', save_flag=False)

    # Compute antagonistic value
    personal_b_val = 2
    reg_dict = compute_aVal(G_hat, personal_b_val, payoff_func, w_prime, loop_vals)

    for k, v in reg_dict.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
