# use this file to construct the game
# then pass this game to the payoff code to construct the finite state machine
import math
import copy
import networkx as nx
import sys
import re
import warnings
import operator

from typing import List, Tuple, Dict, Any
from src.gameconstruction import Graph
from src.compute_payoff import payoff_value
from helper_methods import deprecated

# asserts that this code is tested in linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"


def construct_graph(payoff_func: str, *args, **kwargs) -> Graph:
    G: Graph = Graph(False)
    # create the directed multi-graph
    org_graph = G.create_multigrpah()

    # pattern to dtect exactly 'sup' or 'inf'
    sup_re = re.compile('^sup$')
    inf_re = re.compile('^inf$')

    # TODO: make this more robust my checking limsup and liminf payoff string and throw a warning in case no match
    if inf_re.match(payoff_func):
        gmin = G.construct_Gmin(org_graph)
        G.graph = gmin
    elif sup_re.match(payoff_func):
        gmax = G.construct_Gmax(org_graph)
        G.graph = gmax
    else:
        G.graph = org_graph
    return G


def construct_alt_game(graph: nx.MultiDiGraph, edge: Tuple[str, str]) -> nx.MultiDiGraph:
    """
    A helper method to construct an temporary alternate game without the org edge @edge
    :param graph:
    :param edge:
    :return: A new game graph
    """
    # remove the edge of the form (u, v) from the graph
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(edge[0], edge[1])
    return new_graph


def compute_w_prime(payoff_handle: payoff_value, org_graph: Graph) -> Dict[str, str]:
    print("*****************Constructing W_prime*****************")
    # compute W prime
    # calculate the loop values
    # _loop_vals = payoff_handle.cycle_main()
    # for k, v in _loop_vals.items():
    #     print(f"Play: {k} : val: {v} ")

    # compute the cVal from each node
    w_prime: Dict[str, str] = {}
    for edge in org_graph.graph.edges():
        # if the node belongs to adam, then the corresponding edge is assigned -inf
        if org_graph.graph.nodes(data='player')[edge[0]] == 'adam':
            w_prime.update({str(edge): str(-1 * math.inf)})

        # if the node belongs to eve, then we compute the max cVal from all the possible alternate edges.
        # 1. We construct a new graph without the org edge (u, v)
        # 2. we assign v' (tail node of the alt edge) as the init node
        # 3. Compute the max cVal for all v' from u
        else:
            # get a list of all the cVal for all the alternate edges
            # TODO: check if this is necessary as @tmp_graph is already a deepcopy and I am making a deepcopy of a
            #  deepcopy. Seems redundant but safety over everything.

            # step 1. construct a new game without (u, v)
            tmp_graph = construct_alt_game(org_graph.graph, edge)

            # construct the game without the org edge and find the max from each alternate play
            tmp_cvals = []
            # payoff_handle.graph = tmp_graph
            # payoff_handle.cycle_main()
            # step 2. get all the alt edges (u, v')
            for alt_e in tmp_graph.out_edges(edge[0]):
                # just to be safe, we will work with a copy
                tmp_copied_graph = copy.deepcopy(tmp_graph)
                # construct a new graph with alt_e[1] as the initial vertex and compute loop_vals again and then
                # proceed ahead in this method
                # 1. remove the current init node of the graph
                # 2. add e[1] as the new init vertex
                # 3. compute the loop vals for this new graph
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(alt_e[1])
                tmp_payoff_handle.cycle_main()

                # get all cVal from all the alt edges(v') from a given node (u)
                tmp_cvals.append(tmp_payoff_handle.compute_cVal(alt_e[1]))
            # after going throw all the edges
            if len(tmp_cvals) != 0:
                w_prime.update({str(edge): max(tmp_cvals)})
            # if no edges exist then just compute the cVal from v of the org edge
            else:
                # TODO: check if this correct or not?!
                # make a copy of the org_graph and a another tmp_payoff_handle and compute loop vals
                # payoff_handle.graph = graph.graph
                tmp_copied_graph = copy.deepcopy(org_graph.graph)
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(edge[1])
                tmp_payoff_handle.cycle_main()
                w_prime.update({str(edge): tmp_payoff_handle.compute_cVal(edge[1])})
                # payoff_handle.cycle_main()
                # w_prime.update({edge: payoff_handle.compute_cVal(edge[1])})
                # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())

    print(f"the value of b are {set(w_prime.values())}")

    return w_prime


def new_compute_w_prime_for_g_m(payoff_handle: payoff_value, org_graph: Graph) \
        -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], str]:
    print("*****************Constructing W_prime*****************")
    # compute the cVal from each node
    w_prime: Dict[Tuple[Tuple[str, str], Tuple[str, str]], str] = {}
    for edge in org_graph.graph.edges():
        # if the node belongs to adam, then the corresponding edge is assigned -inf
        if org_graph.graph.nodes(data='player')[edge[0]] == 'adam':
            w_prime.update({edge: str(-1 * math.inf)})

        # if the node belongs to eve, then we compute the max cVal from all the possible alternate edges.
        # 1. We construct a new graph without the org edge (u, v)
        # 2. we assign v' (tail node of the alt edge) as the init node
        # 3. Compute the max cVal for all v' from u
        else:
            # TODO: check if this is necessary as @tmp_graph is already a deepcopy and I am making a deepcopy of a
            #  deepcopy. Seems redundant but safety over everything.
            # step 1. construct a new game without (u, v)
            tmp_graph = construct_alt_game(org_graph.graph, edge)
            # construct the game without the org edge and find the max from each alternate play
            tmp_cvals = []
            # step 2. get all the alt edges (u, v')
            for alt_e in tmp_graph.out_edges(edge[0]):
                # just to be safe, we will work with a copy
                tmp_copied_graph = copy.deepcopy(tmp_graph)
                # construct a new graph with alt_e[1] as the initial vertex and compute loop_vals again and then
                # proceed ahead in this method
                # 1. remove the current init node of the graph
                # 2. add e[1] as the new init vertex
                # 3. compute the loop vals for this new graph
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(alt_e[1])
                tmp_payoff_handle.cycle_main()

                # get all cVal from all the alt edges(v') from a given node (u)
                tmp_cvals.append(tmp_payoff_handle.compute_cVal(alt_e[1]))
            # after going throw all the edges
            if len(tmp_cvals) != 0:
                w_prime.update({edge: max(tmp_cvals)})
            # if no edges exist then just compute the cVal from v of the org edge
            else:
                # TODO: check if this correct or not?!
                # make a copy of the org_graph and a another tmp_payoff_handle and compute loop vals
                # payoff_handle.graph = graph.graph
                tmp_copied_graph = copy.deepcopy(org_graph.graph)
                tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
                tmp_init_node = tmp_payoff_handle.get_init_node()
                # we should ideally only have one init node
                for _n in tmp_init_node:
                    tmp_payoff_handle.remove_attribute(_n, 'init')
                tmp_payoff_handle.set_init_node(edge[1])
                tmp_payoff_handle.cycle_main()
                w_prime.update({edge: tmp_payoff_handle.compute_cVal(edge[1])})
                # payoff_handle.cycle_main()
                # w_prime.update({edge: payoff_handle.compute_cVal(edge[1])})
                # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())

    print(f"the value of b are {set(w_prime.values())}")

    return w_prime

def _construct_g_b_from_g_m(g_hat, org_graph, b, w_prime):
    # each node is dict with the node name as key and 'b' as its value
    g_hat.add_nodes_from([((n), b) for n in org_graph.nodes()])
    # g_hat_node_pattern = re.compile('_')

    # get the init nodes (ideally should only be one) of the org_graph
    init_node = new_get_init_node(org_graph)

    # TODO: In future code add more info to this warning
    assert (len(init_node) == 1), "Detected multiple init nodes in the org graph. This should not be the case"
    # assign each node a player if it hasn't been initialized yet
    for n in g_hat.nodes():
        # assign the nodes of G_b with v1 in it at n[0] to have a 'init' attribute
        if len(n) == 2 and n[0] == init_node[0][0]:
            g_hat.nodes[n]['init'] = True
        if g_hat.nodes(data='player')[n] is None:
            # get the node literal all the way up to "_"
            # start_pos_index_of_underscore = g_hat_node_pattern.search(n).regs[0][0]
            if org_graph.nodes(data='player')[n[0]] == "adam":
                g_hat.nodes[n]['player'] = "adam"
            else:
                g_hat.nodes[n]['player'] = "eve"

    # a sample edge og g_hat: ((".","."),"."),((".","."),".") and
    # a sample edge of org_graph: (".", ""),(".", ".")
    for e in org_graph.edges():
        if w_prime[e] <= b:
            g_hat.add_edge(((e[0]), b), ((e[1]), b))

@deprecated
# def _construct_g_b(g_hat, org_graph, b, w_prime):
#     g_hat.add_nodes_from([f"{n}_{b}" for n in org_graph.nodes()])
#     g_hat_node_pattern = re.compile('_')
#
#     # assign each node a player if it hasn't been initialized yet
#     for n in g_hat.nodes():
#         # assign the nodes of G_b with 1 in it at n[0] to have a 'init' attribute
#         if len(n) > 2 and 'v1' in n:
#             g_hat.nodes[n]['init'] = True
#         if g_hat.nodes(data='player')[n] is None:
#             # get the node literal all the way up to "_"
#             start_pos_index_of_underscore = g_hat_node_pattern.search(n).regs[0][0]
#             if org_graph.nodes(data='player')[n[:start_pos_index_of_underscore]] == "adam":
#                 g_hat.nodes[n]['player'] = "adam"
#             else:
#                 g_hat.nodes[n]['player'] = "eve"
#
#     for e in org_graph.edges():
#         if w_prime[str(e)] <= b:
#             g_hat.add_edge(f"{e[0]}_{b}", f"{e[1]}_{b}")
#     # TODO check is if it efficient to add w_hat here or afterwards by looping over all the edges in G_hat
#     # return graph


def get_max_weight(graph) -> float:
    """
    A helper method to compute the max weight in a given graph
    :param graph:
    :return:
    """

    weight_list = []
    for _, _, weight in graph.edges(data='weight'):
        weight_list.append(weight)

    return float(max(weight_list))


def new_get_init_node(graph: nx.MultiDiGraph) -> List[Tuple[str, str]]:
    # a helper method to find the init node and return
    init_node: List[Tuple[str, str]] = []
    for n in graph.nodes.data("init"):
        if n[1] is True:
            init_node.append(n)
    return init_node


def construct_g_hat_from_g_m(org_graph, w_prime):
    print("*****************Constructing G_hat*****************")
    # construct new graph according to the pseudocode 3
    G_hat = nx.MultiDiGraph(name="G_hat")
    G_hat.add_nodes_from(['v0', 'v1', 'vT'])
    # nodes will be nested tuple from now onwards for ease of implementation
    G_hat.nodes['v0']['player'] = "adam"
    G_hat.nodes['v1']['player'] = "eve"
    G_hat.nodes['vT']['player'] = "eve"

    # add v0 as the initial node
    G_hat.nodes['v0']['init'] = True
    # add the edges with the weights
    G_hat.add_weighted_edges_from(
        [('v0', 'v0', '0'), ('v0', 'v1', '0'), ('vT', 'vT', str(-2 * get_max_weight(org_graph) - 1))])

    # compute the range of w_prime function
    w_set = set(w_prime.values()) - {str(-1 * math.inf)}
    # construct g_b
    for b in w_set:
        _construct_g_b_from_g_m(G_hat, org_graph, b, w_prime)

    # add edges between 1 of G_hat and init(1_b) of graph G_b with edge weights 0
    # get init node of the org graph
    init_node_list: List = new_get_init_node(graph=G_hat)

    def remove_attribute(G, tnode, attr):
        G.nodes[tnode].pop(attr, None)

    # add edge with weigh 0 from v1 to all the init nodes of g_b as (('v1', '1'), '1'); (('v1', '1'), '2'); ....
    for _init_n in init_node_list:
        G_hat.add_weighted_edges_from([('v1', _init_n[0], 0)])
        # remove the init nodes from g_b graph
        if isinstance(_init_n[0], tuple):
            remove_attribute(G_hat, _init_n[0], "init")

    # TODO: looks like a tmp patch; this may or may not work when using sup/inf payoff function.
    #  Need to verify it
    # remove nodes from g_b which has init nodes
    # _init_node = [node[0] for node in G_hat.nodes.data() if node[1].get('init') == True]
    # g_b_node_pattern = re.compile('_')
    # for duplicate_init_node in _init_node:
    #     if g_b_node_pattern.search(duplicate_init_node) is not None:
    #         remove_attribute(G_hat, duplicate_init_node, 'init')

    def w_hat_b(_org_graph: nx.MultiDiGraph, org_edge: Tuple[Tuple[str, str], Tuple[str, str]], b_value: str) -> str:
        """
        an inline function to find the w_prime valued for a g_b graph
        :param _org_graph:
        :param org_edge: edges of the format ("v1", "v2")
        :param b_value:
        :return:
        """
        if float(w_prime[org_edge]) != -1 * math.inf:
            return str(float(w_prime[org_edge]) - float(b_value))
        else:
            try:
                return str(float(_org_graph[org_edge[0]][org_edge[1]][0].get('weight')) - float(b_value))
            except KeyError:
                print(KeyError)
                print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                      "original graph has been modified and the edge {} does not exist".format(org_edge))

    def find_node_before_underscore(_node: str) -> str:
        # start_index = g_b_node_pattern.search(_node).regs[0][0]
        # return _node[:start_index]
        return "comment this"

    def find_b_after_underscore(_node: str) -> str:
        last_index = re.search("_+", _node).regs[0][1]
        return _node[last_index:]

    # add edges with their respective weights; a sample edge ((".","."),"."),((".","."),".")
    for e in G_hat.edges():
        # only add weights if hasn't been initialized
        if G_hat[e[0]][e[1]][0].get('weight') is None:
            # initialize_weights
            # the nodes are stored as string in format "1_1" so we need only the first element
            # condition to check if the node belongs to g_b or not
            # if len(e[0]) > 1:
            # an edge can only exist within a g_b
            assert (e[0][1] == e[1][1]), "Make sure that there only exist edge betwen nodes that belong to the same g_b"
            G_hat[e[0]][e[1]][0]['weight'] = w_hat_b(org_graph, (e[0][0], e[1][0]), e[0][1])

    # for nodes that don't have any outgoing edges add a transition to the terminal node i.e 'T' in our case
    for node in G_hat.nodes():
        if G_hat.out_degree(node) == 0:
            # add transition to the terminal node
            G_hat.add_weighted_edges_from([(node, 'vT', 0)])

    return G_hat


def construct_g_hat(org_graph, w_prime):
    print("*****************Constructing G_hat*****************")
    # construct new graph according to the pseudocode 3
    G_hat = nx.MultiDiGraph(name="G_hat")
    G_hat.add_nodes_from(['v0', 'v1', 'vT'])
    G_hat.nodes['v0']['player'] = "adam"
    G_hat.nodes['v1']['player'] = "eve"
    G_hat.nodes['vT']['player'] = "eve"

    # add v0 as the initial node
    G_hat.nodes['v0']['init'] = True
    # add the edges with the weights
    G_hat.add_weighted_edges_from([('v0', 'v0', '0'), ('v0', 'v1', '0'), ('vT', 'vT', str(-2 * get_max_weight(org_graph) - 1))])

    # compute the range of w_prime function
    w_set = set(w_prime.values()) - {str(-1 * math.inf)}
    # construct g_b
    for b in w_set:
        _construct_g_b_from_g_m(G_hat, org_graph, b, w_prime)

    # add edges between 1 of G_hat and init(1_b) of graph G_b with edge weights 0
    for b in w_set:
        G_hat.add_weighted_edges_from([('v1', f"v1_{b}", 0)])

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

    def w_hat_b(_org_graph: nx.MultiDiGraph, org_edge: Tuple[str, str], b_value) -> str:
        """
        an inline function to find the w_prime valued for a g_b graph
        :param _org_graph:
        :param org_edge: edges of the format "("v1", "v2")"
        :param b_value:
        :return:
        """
        if float(w_prime[str(org_edge)]) != -1 * math.inf:
            return str(float(w_prime[str(org_edge)]) - float(b_value))
        else:
            try:
                return str(float(_org_graph[org_edge[0]][org_edge[1]][0].get('weight')) - float(b_value))
            except KeyError:
                print(KeyError)
                print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                      "original graph has been modified and the edge {} does not exist".format(org_edge))

    def find_node_before_underscore(_node: str) -> str:
        start_index = g_b_node_pattern.search(_node).regs[0][0]
        return _node[:start_index]

    def find_b_after_underscore(_node: str) -> str:
        last_index = re.search("_+", _node).regs[0][1]
        return _node[last_index:]

    # add edges with their respective weights
    for e in G_hat.edges():
        # only add weights if hasn't been initialized
        if G_hat[e[0]][e[1]][0].get('weight') is None:
            # initialize_weights
            # the nodes are stored as string in format "1_1" so we need only the first element
            # condition to check if the node belongs to g_b or not
            if len(e[0]) > 1:
                G_hat[e[0]][e[1]][0]['weight'] = w_hat_b(org_graph, (find_node_before_underscore(e[0]),
                                                                     find_node_before_underscore(e[1])),
                                                         find_b_after_underscore(e[0]))

    # for nodes that don't have any outgoing edges add a transition to the terminal node i.e 'T' in our case
    for node in G_hat.nodes():
        if G_hat.out_degree(node) == 0:
            # add transition to the terminal node
            G_hat.add_weighted_edges_from([(node, 'vT', 0)])

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


def new_compute_aVal_from_g_m(g_hat: nx.MultiDiGraph, meta_b: str, _Val_func: str, w_prime: Dict,
                              org_graph: nx.MultiDiGraph) -> Dict[str, Dict]:
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
                                                              "to w_prime i.e {}.".format(set(w_prime.values()))

    # create a dict which will be update once we find the reg value and strategies for eve and adam
    str_dict = {
        'reg': None,
        'eve': None,
        'adam': None
    }

    # create a empty dict to hold the strategy for eve and adam
    eve_str: Dict[Tuple[Tuple[str, str], str], Tuple[Tuple[str, str], str]] = {}
    adam_str: Dict[Tuple[Tuple[str, str], str], Tuple[Tuple[str, str], str]] = {}

    # update 0 to 1 transition in g_hat and 1 to 1_b depending on the value of b - hyper-paramter chosen by the user
    # NOTE: here the strategy is a dict; key is the current node while the value is the next node.
    #  This would NOT work in condition where there are more than one edge between the same nodes i.e 0-1 with multiple
    #  weights. If that is the case then we need to change the implementation to store the whole edge instead of just
    #  the next node.
    adam_str.update({'v0': "v1"})
    _max_accepted_b: List[str] = []
    for b in set(w_prime.values()) - {str(-1*math.inf)}:
        if b <= meta_b:
            # add all those transition that satisfy the condition
            _max_accepted_b.append(b)

    # get the max_b below the accepted val (meta_b)
    allowed_b: str = max(_max_accepted_b)
    # get the init node
    # get the init nodes (ideally should only be one) of the org_graph
    init_node = new_get_init_node(org_graph)

    # TODO: In future code add more info to this warning
    assert (len(init_node) == 1), "Detected multiple init nodes in the org graph. This should not be the case"
    eve_str.update({"v1": ((init_node[0][0]), allowed_b)})

    def check_node_owner(_node: str) -> bool:
        """
        an inline helper method to check if the node belong to g_b (has "_" in it) or g_hat (no "_" in it)
        :param _node:
        :return: return False if it belongs to g_hat and True if it belongs to g_b
        """
        if re.search("_", _node) is not None:
            return True
        return False

    def find_b_after_underscore(_node: str) -> str:
        last_index = re.search("_+", _node).regs[0][1]
        return _node[last_index:]

    # update strategy for each node
    # 1. adam picks the edge with the min value
    # 2. eve picks the edge with the max value
    for node in g_hat.nodes():
        # I manually add the initial transitions of node 0 and 1 of g_hat graph
        # TODO: change the second condition to <= if you want to compute all the str for all nodes <= @allowed_b
        if isinstance(node, tuple) and float(node[1]) == float(allowed_b):
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
    str_dict['reg'] = -1*float(a_val)

    # update eve and adam str and return it
    str_dict['adam'] = adam_str
    str_dict['eve'] = eve_str

    return str_dict



def compute_aVal(g_hat: nx.MultiDiGraph, meta_b: str, _Val_func: str, w_prime: Dict) -> Dict[str, Dict]:
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
                                                              "to w_prime i.e {}.".format(set(w_prime.values()))

    # create a dict which will be update once we find the reg value and strategies for eve and adam
    str_dict = {
        'reg': None,
        'eve': None,
        'adam': None
    }

    # create a empty dict to hold the strategy for eve and adam
    eve_str: Dict[str, str] = {}
    adam_str: Dict[str, str] = {}

    # update 0 to 1 transition in g_hat and 1 to 1_b depending on the value of b - hyper-paramter chosen by the user
    # NOTE: here the strategy is a dict; key is the current node while the value is the next node.
    #  This would NOT work in condition where there are more than one edge between the same nodes i.e 0-1 with multiple
    #  weights. If that is the case then we need to change the implementation to store the whole edge instead of just
    #  the next node.
    adam_str.update({'v0': "v1"})
    _max_accepted_b: List[str] = []
    for b in set(w_prime.values()) - {str(-1*math.inf)}:
        if b <= meta_b:
            # add all those transition that satisfy the condition
            _max_accepted_b.append(b)

    # get the max_b below the accepted val (meta_b)
    allowed_b: str = max(_max_accepted_b)
    eve_str.update({"v1": f"v1_{allowed_b}"})

    def check_node_owner(_node: str) -> bool:
        """
        an inline helper method to check if the node belong to g_b (has "_" in it) or g_hat (no "_" in it)
        :param _node:
        :return: return False if it belongs to g_hat and True if it belongs to g_b
        """
        if re.search("_", _node) is not None:
            return True
        return False

    def find_b_after_underscore(_node: str) -> str:
        last_index = re.search("_+", _node).regs[0][1]
        return _node[last_index:]

    # update strategy for each node
    # 1. adam picks the edge with the min value
    # 2. eve picks the edge with the max value
    for node in g_hat.nodes():
        # I manually add the initial transitions of node 0 and 1 of g_hat graph
        # TODO: change the second condition to <= if you want to compute all the str for all nodes <= @allowed_b
        if check_node_owner(node) and float(find_b_after_underscore(node)) == float(allowed_b):
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
    str_dict['reg'] = -1*float(a_val)

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
    play = [new_get_init_node(graph)[0][0]]

    # for node in graph.nodes():
    while 1:
        # NOTE: This is assuming that a strategy is deterministic i.e the cardinality of next node is 1
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

def _get_next_node(graph: nx.MultiDiGraph, curr_node: str, func) -> str:
    # return the next node for eve and adam on g_hat game
    # func - either max or min
    assert (func == max or func == min), "Please make sure the deciding function for transitions on the game g_hat for " \
                                         "eve and adam is either max or min"

    wt_list = {}
    for adj_edge in graph.edges(curr_node):
        # get the edge weight, store it in a list and find the max/min and return the next_node
        # G_hat[e[0]][e[1]][0].get('weight')
        wt_list.update({adj_edge: float(graph[adj_edge[0]][adj_edge[1]][0].get('weight'))})
        # wt_list.append(graph[adj_edge[0]][adj_edge[1]][0].get('weight'))

    next_node: str = func(wt_list.items(), key=operator.itemgetter(1))[0]

    return next_node[1]


def main():
    payoff_func = "sup"
    print(f"*****************Using {payoff_func}*****************")
    # construct graph
    graph = construct_graph(payoff_func)
    p = payoff_value(graph.graph, payoff_func)

    # FIXME: fails when using inf/sup payoff function
    # construct W prime
    w_prime = new_compute_w_prime_for_g_m(p, graph)

    # construct G_hat
    G_hat = construct_g_hat_from_g_m(graph.graph, w_prime)

    # use methods from the Graph class create a visualization
    plot_graph(G_hat, file_name='src/config/g_hat_graph', save_flag=False)

    # Compute antagonistic value
    personal_b_val = '2'
    reg_dict = new_compute_aVal_from_g_m(G_hat, personal_b_val, payoff_func, w_prime, graph.graph)

    for k, v in reg_dict.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
