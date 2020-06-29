# use this file to construct the game
# then pass this game to the payoff code to construct the finite state machine
import math
import copy
import networkx as nx
import sys
import re
import warnings
import random

from typing import List, Tuple, Dict
from src.compute_payoff import payoff_value
from src.graph.graph import GraphFactory
from src.graph.graph import TwoPlayerGraph, ProductAutomaton

# asserts that this code is tested in linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"

"""
acc_max_edge_weight : flag for graph with accepting states that have a self loop with max weight(W). In this case except 
for g_bmax all accepting states that belong to g_{b\b_max} will have an edge from acc_state to vT. So with this flag I 
manually remove all those edges and instead add a self-loop with edge weight of zero in g_hat.
"""
acc_max_edge_weight = False
# flag to use accepting_states_least_edge_weight
"""
acc_min_edge_weight : flag for graph with accepting states that have a self loop with edge weight 0. In this case all 
the accepting states in all the g_b in g_hat DO NOT transit to vT but they have a self weight of -b 
(w(e) - b => 0 - b => -b). Thus by default a transition to acc_state does NOT result in zero regret except for when 
b = 0.0. With this flag we manually modify the edge weight for all the accepting states to be zero.
"""
acc_min_edge_weight = False
# flag to add biasing to choose the final strategy with the accepting state in it
"""
choose_acc_state_run : Say we have multiple runs with zero regret, then we manually check if a play contains the 
acc_state(s). If it does then we return the corresponding strategy. 

"""
choose_acc_state_run = False
# print strategy even if there does not exist a non-zero regret on g_hat
"""
bypass_implementation : a flag to visualize a strategy even is there does not exist a non-zero regret because according
to the paper, if there does not exist a non-zero regret then adam does not proceed from v0. This flag forces adam to 
play v0 to v1.
"""
bypass_implementation = True


def construct_graph(payoff_func: str, debug: bool = False, use_alias: bool = False,
                    scLTL_formula: str = "", plot: bool = False, prune: bool = False) -> ProductAutomaton:
    """
    A helper method to construct a graph using the GraphFactory() class, given a boolean formula
    :param payoff_func: A payoff function
    :param debug :
    :param use_alias
    :param scLTL_formula
    :param plot
    :return: Return graph G or Gmin/Gmax if payoff function is inf/sup respectively.
    """
    print("=========================================")
    print(f"Using scLTL formula : {scLTL_formula}")
    print("=========================================")
    # pattern to detect exactly 'sup' or 'inf'
    sup_re = re.compile('^sup$')
    inf_re = re.compile('^inf$')

    if inf_re.match(payoff_func):
        _graph = GraphFactory._construct_gmin_graph(debug=debug, plot=plot, prune=prune)
    elif sup_re.match(payoff_func):
        _graph = GraphFactory._construct_gmax_graph(debug=debug, plot=plot, prune=prune)
    else:
        _graph = GraphFactory._construct_product_automaton_graph(use_alias=use_alias, scLTL_formula=scLTL_formula,
                                                                 plot=plot, debug=debug, prune=prune)
    return _graph


def construct_alt_game(graph: nx.MultiDiGraph, edge: Tuple[Tuple, Tuple]) -> nx.MultiDiGraph:
    """
    A helper method to construct an temporary alternate game without the org edge @edge
    :param graph: graph org_graph
    :param edge: a tuple of nodes of the form (u, v)
    :return: A new game graph without the edge (u, v)
    """
    # remove the edge (u, v) from the graph
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(edge[0], edge[1])
    return new_graph


def _compute_max_cval_from_v(graph: nx.MultiDiGraph, payoff_handle: payoff_value, node: Tuple):
    """
    A helper method to compute the cVal from a given vertex (@node) for a give graph @graph
    :param graph: The graph on which would like to compute the cVal
    :param payoff_handle: instance of the @compute_value() to compute the cVal
    :param node: The node from which we would like to compute the cVal
    :return: returns a single max value. If multiple plays have the max_value then the very first occurance is returned
    """
    tmp_copied_graph = copy.deepcopy(graph)
    # construct a new graph with node as the initial vertex and compute loop_vals again
    # 1. remove the current init node of the graph
    # 2. add @node as the new init vertex
    # 3. compute the loop-vals for this new graph
    tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
    # FIXME: even if you are not dealing with a new init node, this process finds the current init node, deletes it and
    #  adds the same thing again. Looks redundant!
    tmp_init_node = tmp_payoff_handle.get_init_node()
    # we should ideally only have one init node
    for _n in tmp_init_node:
        tmp_payoff_handle.remove_attribute(_n, 'init')
    tmp_payoff_handle.set_init_node(node)
    tmp_payoff_handle.cycle_main()

    return tmp_payoff_handle.compute_cVal(node)


def compute_w_prime(payoff_handle: payoff_value, org_graph: nx.MultiDiGraph) \
        -> Dict[Tuple, str]:
    """
    A method to compute w_prime function based on Algo 2. pseudocode. This function is a mapping from each edge to a
    real valued number.
    :param payoff_handle: instance of the @compute_value() to compute the cVal
    :param org_graph: The orginal graph from which we compute the mapping for all the edges in this graph.
    :return: A dict mapping each edge (tuple of nodes(tuple)) to a finite value
    """
    print("*****************Constructing W_prime*****************")
    # compute the cVal from each node
    w_prime: Dict[Tuple, str] = {}
    for edge in org_graph.edges():
        # if the node belongs to adam, then the corresponding edge is assigned -inf
        if org_graph.nodes(data='player')[edge[0]] == 'adam':
            w_prime.update({edge: -1 * math.inf})

        # if the node belongs to eve, then we compute the max cVal from all the possible alternate edges.
        # 1. We construct a new graph without the org edge (u, v)
        # 2. we assign v' (tail node of the alt edge) as the init node
        # 3. Compute the max cVal for all v' from u
        else:
            # TODO: check if this is necessary as @tmp_graph is already a deepcopy and I am making a deepcopy of a
            #  deepcopy. Seems redundant but safety over everything.
            # step 1. construct a new game without (u, v)
            tmp_graph = construct_alt_game(org_graph, edge)
            # construct the game without the org edge and find the max from each alternate play
            tmp_cvals = []
            # step 2. get all the alt edges (u, v')
            for alt_e in tmp_graph.out_edges(edge[0]):
                # get all cVal from all the alt edges(v') from a given node (u)
                tmp_cvals.append(_compute_max_cval_from_v(tmp_graph, payoff_handle, alt_e[1]))
            # after going throw all the alternate edges
            if len(tmp_cvals) != 0:
                w_prime.update({edge: max(tmp_cvals)})
            # if no alternate edges exist then just compute the cVal from v of the org edge
            else:
                # make a copy of the org_graph and a another tmp_payoff_handle and compute loop vals
                w_prime.update({edge: _compute_max_cval_from_v(org_graph, payoff_handle, edge[1])})

    print(f"the value of b are {set(w_prime.values())}")

    return w_prime


def _construct_g_b(g_hat: nx.MultiDiGraph, org_graph: nx.MultiDiGraph, b, w_prime: Dict,
                   init_node: List[Tuple], accp_node: List[Tuple]) -> None:
    # each node is dict with the node name as key and 'b' as its value
    g_hat.add_nodes_from([((n), b) for n in org_graph.nodes()])
    # TODO: Instead of computing init nodes every time from scratch pass them as an argument to this function
    # # get the init nodes (ideally should only be one) of the org_graph
    # init_node = get_init_node(org_graph)

    assert (len(init_node) == 1), f"Detected multiple init nodes in the org graph: {[n for n in init_node]}. " \
                                  f"This should not be the case"
    # assign each node a player if it hasn't been initialized yet
    for n in g_hat.nodes():
        # assign the nodes of G_b with v1 in it at n[0] to have a 'init' attribute
        if len(n) == 2 and n[0] == init_node[0][0]:
            g_hat.nodes[n]['init'] = True
        # assign the nodes of G_b with 'accepting' attribute
        for _accp_n in accp_node:
            if len(n) == 2 and n[0] == _accp_n[0]:
                g_hat.nodes[n]['accepting'] = True
        if g_hat.nodes(data='player')[n] is None:
            if org_graph.nodes(data='player')[n[0]] == "adam":
                g_hat.nodes[n]['player'] = "adam"
            else:
                g_hat.nodes[n]['player'] = "eve"

    # a sample edge og g_hat: ((".","."),"."),((".","."),".") and
    # a sample edge of org_graph: (".", ""),(".", ".")
    for e in org_graph.edges():
        if float(w_prime[e]) <= float(b):
            g_hat.add_edge(((e[0]), b), ((e[1]), b))


def get_max_weight(graph: nx.MultiDiGraph) -> float:
    """
    A helper method to compute the max weight in a given graph
    :param graph:
    :return:
    """

    weight_list = []
    for _, _, weight in graph.edges(data='weight'):
        weight_list.append(weight)

    return float(max(weight_list))


def get_accepting_state_node(graph: nx.MultiDiGraph) -> List[Tuple]:
    # a helper method to find the set of accepting nodes
    accpeting_nodes: List[Tuple] = []
    for n in graph.nodes.data("accepting"):
        if n[1] is True:
            accpeting_nodes.append(n)

    return accpeting_nodes


def get_init_node(graph: nx.MultiDiGraph) -> List[Tuple]:
    # a helper method to find the init node and return.
    init_node: List[Tuple[str, str]] = []
    for n in graph.nodes.data("init"):
        if n[1] is True:
            init_node.append(n)

    # init node is of the form ((node, bool value)). To access the node we need to do [0][0]
    return init_node


def construct_g_hat(org_graph: nx.MultiDiGraph, w_prime: Dict[Tuple, str]) -> TwoPlayerGraph:
    print("*****************Constructing G_hat*****************")
    # construct new graph according to the pseudocode 3
    # G_hat: nx.MultiDiGraph = nx.MultiDiGraph(name="G_hat")
    G_hat = TwoPlayerGraph(graph_name="G_hat", config_yaml="config/G_hat", save_flag=False)
    G_hat.construct_graph()
    G_hat.add_states_from(['v0', 'v1', 'vT'])
    # G_hat.add_nodes_from(['v0', 'v1', 'vT'])

    # G_hat.nodes['v0']['player'] = "adam"
    # G_hat.nodes['v1']['player'] = "eve"
    # G_hat.nodes['vT']['player'] = "eve"

    G_hat.add_state_attribute('v0', 'player', 'adam')
    G_hat.add_state_attribute('v1', 'player', 'eve')
    G_hat.add_state_attribute('vT', 'player', 'eve')

    # add v0 as the initial node
    # G_hat.nodes['v0']['init'] = True
    G_hat.add_initial_state('v0')

    # add accepting states to g_hat
    accp_nodes = get_accepting_state_node(org_graph)
    # add the edges with the weights
    G_hat.add_weighted_edges_from([
        ('v0', 'v0', '0'), ('v0', 'v1', '0'), ('vT', 'vT', str(-2 * get_max_weight(org_graph) - 1))])
    # G_hat.add_weighted_edges_from(
    #     [('v0', 'v0', '0'), ('v0', 'v1', '0'), ('vT', 'vT', str(-2 * get_max_weight(org_graph) - 1))])

    # compute the range of w_prime function
    w_set = set(w_prime.values()) - {-1 * math.inf}
    org_init_nodes = get_init_node(org_graph)

    # construct g_b
    for b in w_set:
        _construct_g_b(G_hat._graph, org_graph, b, w_prime, org_init_nodes, accp_nodes)

    # add edges between v1 of G_hat and init nodes(v1_b/ ((v1, 1), b) of graph G_b with edge weights 0
    # get init node of the org graph
    init_node_list: List[Tuple] = get_init_node(graph=G_hat._graph)

    def remove_attribute(G, tnode, attr):
        G.nodes[tnode].pop(attr, None)

    # add edge with weigh 0 from v1 to all the init nodes of g_b as (('v1', '1'), '1'); (('v1', '1'), '2'); ....
    for _init_n in init_node_list:
        if isinstance(_init_n[0], tuple):
            G_hat.add_weighted_edges_from([('v1', _init_n[0], 0)])
            remove_attribute(G_hat._graph, _init_n[0], "init")

    def w_hat_b(_org_graph: nx.MultiDiGraph, org_edge: Tuple[Tuple[str, str], Tuple[str, str]], b_value: str) -> str:
        """
        an inline function to find the w_hat value for a g_b graph
        :param _org_graph:
        :param org_edge: edges of the format ("v1", "v2") or Tuple of tuples
        :param b_value:
        :return:
        """
        # if float(w_prime[org_edge]) != -1 * math.inf:
        #     return str(float(_org_graph[org_edge[0]][org_edge[1]][0].get('weight')) - float(b_value))
        # else:
        try:
            return str(float(_org_graph[org_edge[0]][org_edge[1]][0].get('weight')) - float(b_value))
        except KeyError:
            print(KeyError)
            print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                  "original graph has been modified and the edge {} does not exist".format(org_edge))

    # add edges with their respective weights; a sample edge ((".","."),"."),((".","."),".") for with gmin/gmax and
    # with ((".","."),(".", "."))
    for e in G_hat._graph.edges():
        # only add weights if hasn't been initialized
        if G_hat._graph[e[0]][e[1]][0].get('weight') is None:
            # an edge can only exist within a graph g_b
            assert (e[0][1] == e[1][1]), \
                "Make sure that there only exist edge between nodes that belong to the same g_b"
            if acc_min_edge_weight and G_hat._graph.nodes[e[0]].get('accepting') is not None:
                G_hat._graph[e[0]][e[1]][0]['weight'] = '0'
            else:
                G_hat._graph[e[0]][e[1]][0]['weight'] = w_hat_b(org_graph, (e[0][0], e[1][0]), e[0][1])

    # for nodes that don't have any outgoing edges add a transition to the terminal node i.e 'T' in our case
    for node in G_hat._graph.nodes():
        if G_hat._graph.out_degree(node) == 0:
            if acc_max_edge_weight:
                # if the node belongs to the accepting state then add a self-loop to itself
                if G_hat._graph.nodes[node].get('accepting') is not None:
                    G_hat.add_weighted_edges_from([(node, node, 0)])
                    continue
            # add transition to the terminal node
            G_hat.add_weighted_edges_from([(node, 'vT', 0)])

    return G_hat


def plot_graph(graph: nx.MultiDiGraph, file_name: str, save_flag: bool = True, visualize_str: bool = False,
               combined_strategy: Dict[Tuple, Tuple] = None, plot=False) -> None:
    """
    A helper method to plot a given graph and save it if the @save_flag is True.
    :param graph: The graph to be plotted
    :param file_name: The name of the yaml file to be dumped in the /config directory. The graph is saved as a plot
    based on the name of the graph (graph.name attribute) in the /graph directory.
    :param save_flag: flag to save the plot. If False then the plots don't show up and are not saved  as well.
    """
    print(f"*****************Plotting graph with save_flag = {save_flag}*****************")
    if visualize_str:
        _add_strategy_flag(graph, combined_strategy)

    # grate a two_player_game using g_hat graph and plot the graph
    GraphFactory.get_two_player_game(graph, "config/g_hat_graph", file_name, save_flag=save_flag, plot=plot)

def _check_non_zero_regret(graph_g_hat: nx.MultiDiGraph, org_graph: nx.MultiDiGraph,  w_prime, value_func,
                           str_dict) -> Tuple[Dict[float, Dict], bool]:
    """
    A helper method to check if there exist non-zero regret in the game g_hat. If yes return true else False
    :param graph_g_hat: graph g_hat on which we would like to compute the regret value
    :param w_prime: the set of bs
    :return: A Tuple cflag; True if Reg > 0 else False
    """
    # get init_node of the original graph
    init_node = get_init_node(org_graph)

    for b in set(w_prime.values()) - {-1 * math.inf}:
        _eve_str: Dict[Tuple, Tuple] = {}
        _adam_str: Dict[Tuple, Tuple] = {}
        # assume adam takes v0 to v1 edge - accordingly update the strategy
        _adam_str.update({"v0": "v1"})
        # add this transition to the strategy of eve and then play in the respective copy of G_b
        _eve_str.update({'v1': ((init_node[0][0]), b)})
        _eve_str.update({'vT': 'vT'})
        for node in graph_g_hat.nodes():
            # I manually add the initial transitions of node 0 and 1 of g_hat graph
            if isinstance(node, tuple) and float(node[1]) == float(b):
                # if the node belongs to adam
                if graph_g_hat.nodes[node]['player'] == 'adam':
                    # get the next node and update
                    _adam_str.update({node: _get_next_node(graph_g_hat, node, min)})
                # if node belongs to eve
                elif graph_g_hat.nodes[node]['player'] == 'eve':
                    _eve_str.update({node: _get_next_node(graph_g_hat, node, max)})
                else:
                    raise warnings.warn(f"The node {node} does not belong either to eve or adam. This should have "
                                        f"never happened")
        # update str_dict
        str_dict.update({b: {'eve': _eve_str}})
        str_dict[b].update({'adam': _adam_str})
        # compute the regret value for this b
        plays: List[List[Tuple]] = _compute_all_plays(graph_g_hat, {**_eve_str, **_adam_str})
        # if there is only possible play then that means the str we have is already deterministic
        if len(plays) == 1:
            reg = -1 * float(_play_loop(graph_g_hat, plays[0], value_func))
        else:
            reg, _adam_str, _eve_str = _find_optimal_str(plays, graph_g_hat, value_func, _adam_str, _eve_str)
            str_dict[b]['eve'] = _eve_str
            str_dict[b]['adam'] = _adam_str

        str_dict[b].update({'reg': reg})
        if reg > 0:
            return str_dict, True

    return str_dict, False


def _find_optimal_str(plays: List[List[Tuple]], graph_g_hat: nx.MultiDiGraph, value_func: str,
                      adam_str: Dict[Tuple, Tuple],
                      eve_str: Dict[Tuple, Tuple]) -> Tuple[float, Dict, Dict]:
    """
    A helper method to find the most optimal strategy in a non-deterministic strategy
    :param plays: a list of plays(tuple)
    :param graph_g_hat: the graph on which we will roll out these strategies to compute the Val of the play
    :param value_func: the value function to determine a finite quantity of an infinite play (e.f sup, mean, liminf)
    :param adam_str: A dict mapping each node that belongs to adam to the next node
    :param eve_str: A dict mapping each node that belongs to eve to the next node
    :return: A tuple of the min reg value, the corresponding DETERMINISTIC strategy for eve and adam
    """

    def __check_acc_node(play: List[Tuple], accpeting_state) -> Tuple[bool, List[Tuple]]:
        # an inline function to help check is a play contains the accepting state or not
        # if yes then return True else False
        for state in accpeting_state:
            if state[0] in play:
                return True, play
        return False, None

    # get accepting state(s)
    accp_nodes = get_accepting_state_node(graph_g_hat)
    flag = False
    if choose_acc_state_run:
        for play in plays:
            flag, play = __check_acc_node(play, accp_nodes)
            if flag:
                min_reg_val = -1 * float(_play_loop(graph_g_hat, play, value_func))
                min_play = play
                break

    if not flag:
        # a temp regret dict of format {play: reg_value}
        tmp_reg_dict = {}
        for play in plays:
                tmp_reg_dict.update({tuple(play): -1 * float(_play_loop(graph_g_hat, play, value_func))})

        # after finding the strategy with the least reg value determinize the strategy
        min_reg_val = min(tmp_reg_dict.values())
        min_play = random.choice([k for k in tmp_reg_dict if tmp_reg_dict[k] == min_reg_val])

    # update the strategy
    for inx, node in enumerate(min_play):
        # only proceed up to to the second last node
        if inx < len(min_play) - 1:
            # if the node belongs to eve then update eve strategy
            if graph_g_hat.nodes[node]['player'] == 'adam':
                # get the next node and update the strategy
                adam_str[node] = min_play[inx + 1]
            # if the node belongs to eve then update eve strategy
            else:
                eve_str[node] = min_play[inx + 1]

    return min_reg_val, adam_str, eve_str


def compute_aVal(g_hat: nx.MultiDiGraph, _Val_func: str, w_prime: Dict, org_graph: nx.MultiDiGraph) \
        -> Dict[str, Dict]:
    """
    A function to compute the regret value according to algorithm 4 : Reg = -1 * Val(.,.) on g_hat
    :param g_hat: a directed multi-graph constructed using construct_g_hat()
    :param Val: a payoff function that belong to {limsup, liminf, sup, inf}
    :return: a dict consisting of the reg value and strategy for eve and adam respectively
    """
    print(f"*****************Computing regret and strategies for eve and adam*****************")
    # assert (float(meta_b) in set(w_prime.values()) - {-1*math.inf}), "make sure that the b value manually
    # entered belongs to w_prime i.e {}.".format(set(w_prime.values()))
    # create a dict which will be update once we find the reg value and strategies for eve and adam
    # The str dict looks like this
    """
    str_dict = {
        b: {
            'reg': None,
            'eve': None,
            'adam': None,
        }
    }
    """
    final_str_dict = {}
    str_dict = {}

    # check if you adam can ensure non-zero regret
    str_dict, reg_flag = _check_non_zero_regret(graph_g_hat=g_hat, org_graph=org_graph, w_prime=w_prime, 
                                                value_func=_Val_func, str_dict=str_dict)
    if reg_flag:
        print("A non-zero regret exists and thus adam will play from v0 to v1 in g_hat")
    else:
        print("A non-zero regret does NOT exist and thus adam will play v0 to v0")
        if bypass_implementation:
            # after computing all the reg value find the str with the least reg
            min_reg_b = min(str_dict, key=lambda key: str_dict[key]['reg'])

            # return the corresponding str and reg value
            final_str_dict.update({'reg': str_dict[min_reg_b]['reg']})
            final_str_dict.update({'eve': str_dict[min_reg_b]['eve']})
            final_str_dict.update({'adam': str_dict[min_reg_b]['adam']})
            return final_str_dict
        else:
            return {}

    # update 0 to 1 transition in g_hat and 1 to 1_b depending on the value of b - hyper-paramter chosen by the user
    # NOTE: here the strategy is a dict; key is the current node while the value is the next node.
    # get the init nodes (ideally should only be one) of the org_graph
    init_node = get_init_node(org_graph)
    assert (len(init_node) == 1), f"Detected multiple init nodes in the org graph: {[n for n in init_node]}. " \
                                  f"This should not be the case"

    reg_threshold: str = input(f"Enter a value of threshold with the range: "
                               f"[0, {-1*(-2 * get_max_weight(org_graph) - 1)}]: \n")

    try:
        assert 0 <= float(reg_threshold) <= -1*(-2 * get_max_weight(org_graph) - 1), "please enter a valid value " \
                                                                                     "within the above range"
    except:
        reg_threshold = input(
            f"Enter a value of threshold with the range: [0, {-1 * (-2 * get_max_weight(org_graph) - 1)}]: \n")

    # update strategy for each node
    # 1. adam picks the edge with the min value
    # 2. eve picks the edge with the max value
    for b in set(w_prime.values()) - {-1 * math.inf}:
        # if we haven't computed a strategy for this b value then proceed ahead
        if str_dict.get(b) is None:
            eve_str: Dict[Tuple, Tuple] = {}
            adam_str: Dict[Tuple, Tuple] = {}
            # update adam's strategy from v0 to v1 and eve's strategy from v1 to (vI, ,b)
            adam_str.update({"v0": "v1"})
            eve_str.update({'v1': ((init_node[0][0]), b)})
            eve_str.update({'vT': 'vT'})
            for node in g_hat.nodes():
                # I manually add the initial transitions of node 0 and 1 of g_hat graph
                if isinstance(node, tuple) and float(node[1]) == float(b):
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

            # update the str dict
            str_dict.update({b: {'eve': eve_str}})
            str_dict[b].update({'adam': adam_str})

            plays: List[List[Tuple]] = _compute_all_plays(g_hat, {**eve_str, **adam_str})
            # if there is only possible play then that means the str we have is already deterministic
            if len(plays) == 1:
                reg = -1 * float(_play_loop(g_hat, plays[0], _Val_func))
            else:
                reg, adam_str, eve_str = _find_optimal_str(plays, g_hat, _Val_func, adam_str, eve_str)
                str_dict[b]['eve'] = eve_str
                str_dict[b]['adam'] = adam_str

            str_dict[b].update({'reg': reg})

            # compute the reg value and update the str_dict respectively
            # reg = -1 * float(_play_loop(g_hat, {**eve_str, **adam_str}, _Val_func))
            # str_dict[b].update({'reg': reg})

    # create a tmp dict of strategies that have reg <= reg_threshold
    __tmp_dict = {}
    for k, v in str_dict.items():
        if v['reg'] <= float(reg_threshold):
            __tmp_dict.update({k: str_dict[k]})

    if len(list(__tmp_dict.keys())) == 0:
        print(f"There does not exist any strategy within the given threshold: {reg_threshold}")
        return {}

    # after computing all the reg value find the str with the least reg
    min_reg_b = min(__tmp_dict, key=lambda key: __tmp_dict[key]['reg'])

    # return the corresponding str and reg value
    final_str_dict.update({'reg': __tmp_dict[min_reg_b]['reg']})
    final_str_dict.update({'eve': __tmp_dict[min_reg_b]['eve']})
    final_str_dict.update({'adam': __tmp_dict[min_reg_b]['adam']})

    return final_str_dict

def _add_strategy_flag(graph: nx.MultiDiGraph, strategy: Dict[Tuple, Tuple]) -> None:
    """
    A helper method to add an attribute/flag which makes it easier to visualize the
    :param graph:
    :param strategy:
    """
    # add strategy as an attribute for plotting the final strategy
    nx.set_edge_attributes(graph, False, 'strategy')

    for curr_node, next_node in strategy.items():
        if isinstance(next_node, list):
            for n_node in next_node:
                graph.edges[curr_node, n_node, 0]['strategy'] = True
        else:
            graph.edges[curr_node, next_node, 0]['strategy'] = True


def _compute_all_plays(graph: nx.MultiDiGraph, strategy: Dict[Tuple, Tuple]) -> List[List[Tuple]]:
    """
    A helper method to compute all the plays possible given a strategy
    :return:
    """
    play_lst = []
    play = [get_init_node(graph)[0][0]]
    for n in play:
        _compute_all_plays_utils(n, play, strategy, graph, play_lst)

    return play_lst


def _compute_all_plays_utils(n, play, strategy: Dict[Tuple, List], graph: nx.MultiDiGraph, play_lst) -> List[Tuple]:
    if not isinstance(strategy[n], list):
        play.append(strategy[n])
        if play.count(play[-1]) >= 2:
            # print(play)
            play_lst.append(play)
    else:
        for node in strategy[n]:
            # path = []
            path = copy.deepcopy(play)
            path.append(node)
            if path.count(path[-1]) < 2:
                _compute_all_plays_utils(node, path, strategy, graph, play_lst)
                # print(path)
            else:
                # print(path)
                play_lst.append(path)


def _play_loop(graph: nx.MultiDiGraph, play: List[Tuple], payoff_func: str) -> str:
    """
    helper method to compute the loop value for a given payoff function
    :param graph: graph g_hat
    :param strategy: A mapping from a each node of g_hat to the next node
    :return: The value of the loop when following the strategy @strategy
    """
    # # add nodes to this stack and as soon as a loop is found we break
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
    # str_graph.add_weighted_edges_from([(play[-1],
    #                                     play[-2],
    #                                     graph[play[-1]][play[-2]][0].get('weight'))])

    # add init node
    str_graph.nodes[play[0]]['init'] = True
    # add this graph to compute_payoff class
    tmp_p_handle = payoff_value(str_graph, payoff_func)
    _loop_vals = tmp_p_handle.cycle_main()
    play_key = tuple(tmp_p_handle._convert_stack_to_play_str(play))

    return _loop_vals[play_key]


def _get_next_node(graph: nx.MultiDiGraph, curr_node: Tuple, func) -> List[Tuple]:
    assert (func == max or func == min), "Please make sure the deciding function for transitions on the game g_hat for " \
                                         "eve and adam is either max or min"
    # NOTE: if there are multiple edges with same weight, it select the first one with the min/max value.
    #  Thus |next_node[1]| is 1.
    wt_list = {}
    for adj_edge in graph.edges(curr_node):
        # get the edge weight, store it in a list and find the max/min and return the next_node
        wt_list.update({adj_edge: float(graph[adj_edge[0]][adj_edge[1]][0].get('weight'))})

    min_value = func(wt_list.values())
    next_nodes: List[Tuple] = [k[1] for k in wt_list if wt_list[k] == min_value]

    return next_nodes

def map_g_hat_str_to_org_graph(g_hat: nx.MultiDiGraph, org_graph: TwoPlayerGraph, strategy: Dict) -> Dict:
    #  A function to map the strategy from g_hat to the original strategy

    # a list to keep track of nodes visited
    org_strategy = []
    g_hat_strategy = []

    g_hat_tmp = {}
    org_tmp = {}

    # start from the initial node of the org_graph
    curr_node = get_init_node(g_hat)[0][0] # v1
    g_hat_strategy.append(curr_node)  # [v1]

    # get the value associated with this key
    next_node = strategy[curr_node]  # v2
    g_hat_tmp.update({curr_node: next_node})
    # org_strategy.append(next_node)
    while next_node not in g_hat_strategy:
        g_hat_strategy.append(next_node)  # [v1, v2, ... ]
        curr_node = next_node  # curr_node = v2
        next_node = strategy[curr_node]  # v3
        g_hat_tmp.update({curr_node: next_node})

    # check if the current node in g_hat_strategy does exist in org_graph. If so add it to org_strategy
    for node in g_hat_strategy:
        if node != "v0" and node != "v1":
            if org_graph._graph_name == "Gmin_graph" or org_graph._graph_name == "Gmax_graph":
                if org_graph._trans_sys._graph.has_node(node[0][0][0]):
                    org_strategy.append(node[0][0][0])
            else:
                if org_graph._trans_sys._graph.has_node(node[0][0]):
                    org_strategy.append(node[0][0])

    for u_node, v_node in g_hat_tmp.items():
        if u_node != "v0" and u_node != "v1":
            if org_graph._graph_name == "Gmin_graph" or org_graph._graph_name == "Gmax_graph":
                if org_graph._trans_sys._graph.has_node(u_node[0][0][0]):
                    org_tmp.update({u_node[0][0][0]: v_node[0][0][0]})
            else:
                if org_graph._trans_sys._graph.has_node(u_node[0][0]):
                    org_tmp.update({u_node[0][0]: v_node[0][0]})

    return org_tmp


def main():
    payoff_func = "limsup"
    print(f"*****************Using {payoff_func}*****************")

    # construct graph
    prod_graph = construct_graph(payoff_func, scLTL_formula="!b U (!a U c)", plot=True, debug=True, prune=True)
    p = payoff_value(prod_graph._graph, payoff_func)

    # construct W prime
    w_prime = compute_w_prime(p, prod_graph._graph)

    # construct G_hat
    G_hat = construct_g_hat(prod_graph._graph, w_prime)

    # NOTE: The strategy that eve comes up with is the strategy with the least regret.
    #  The regret value should be within [0, -2W - 1]; W = Max weight in the original graph
    #  Adam plays from v0 to v1-only if he can ensure a non-zero regret (the Val of the corresponding play in
    #  g_hat should be > 0)
    #  Eve selects the strategy with the least regret (below the given threshold)
    reg_dict = compute_aVal(G_hat._graph, payoff_func, w_prime, prod_graph._graph)

    if len(list(reg_dict.keys())) != 0:
        for k, v in reg_dict.items():
            print(f"{k}: {v}")

    # visualize the strategy
    plot_graph(G_hat._graph, file_name='config/g_hat_graph',
               save_flag=True,
               visualize_str=True,
               combined_strategy={**reg_dict['eve'], **reg_dict['adam']}, plot=True)

    # map back strategy from g_hat to the original graph
    org_strategy = map_g_hat_str_to_org_graph(G_hat._graph, prod_graph, {**reg_dict['eve'], **reg_dict['adam']})

    plot_graph(prod_graph._trans_sys._graph, file_name="config/trans_sys",
               save_flag=True,
               visualize_str=True,
               combined_strategy=org_strategy, plot=True)

if __name__ == "__main__":
    main()
