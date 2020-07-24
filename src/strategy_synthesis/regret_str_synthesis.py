import math
import copy
import warnings
import random
import sys

from _collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from src.graph import Graph, graph_factory
from src.graph import TwoPlayerGraph
from src.graph import ProductAutomaton
from src.payoff import Payoff, payoff_factory
from helper_methods import deprecated

# asserts that this code is tested in linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"


class RegretMinimizationStrategySynthesis:
    """
    This class implements Algo 1, 2, 3, and 4 as sepcified the pseudocode pdf file

    :param      graph:          A concrete instance of class TwoPlayerGraph or ProductAutomation
                                (depending on how you construct the Graph) on which we will be performing the
                                regret minimizing strategy synthesis
    :param      payoff          A concrete instance of the payoff function to be used. Currently the code has
                                inf, sup, liminf, limsup, mean and Cumulative payoff implementation.
    """
    def __init__(self,
                 graph: Graph,
                 payoff: Payoff) -> 'RegretMinimizationStrategySynthesis':
        self.graph = graph
        self.payoff = payoff

    def compute_W_prime_finite(self):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
        alternate strategy game. This is slightly different in finite payoffs in that we add that edge (u,v')
        that we skip in infinite payoff computation. In that case its fine, as it's the edges that we encounter
        infinitely often. But, for finite payoff computation skipping edge will effect the W_prime. Computation

        Changes : We manually add the edge weight associated with (u, v') that we skip.
        """

        print("*****************Constructing W_prime*****************")
        coop_dict = self._compute_cval_finite()

        w_prime: Dict[Tuple: str] = {}

        for edge in self.graph._graph.edges():

            # if the node belongs to adam, then the corresponding edge is assigned -inf
            if self.graph._graph.nodes(data='player')[edge[0]] == 'adam':
                w_prime.update({edge: -1 * math.inf})

            else:
                # a list to save all the alternate strategy cVals from a node and then selecting
                # the max of it
                tmp_cvals = []
                out_going_edge = set(self.graph._graph.out_edges(edge[0])) - set([edge])
                for alt_e in out_going_edge:
                    # get the edge weight associated with this edge
                    curr_w = self.graph.get_edge_weight(alt_e[0],
                                               alt_e[1])

                    # if the coop_dict == inf or -inf then we don't care
                    if (coop_dict[alt_e[1]][0] == -math.inf) or (coop_dict[alt_e[1]][0] == math.inf):
                        tmp_cvals.append(coop_dict[alt_e[1]][0])
                    # if coop_dict is a finite value, then check if the current edge was already encountered or not
                    else:
                        if self.__check_edge_in_cval_play(alt_e, coop_dict[edge[1]][1]):
                            tmp_cvals.append(coop_dict[alt_e[1]][0])
                        else:
                            tmp_cvals.append(coop_dict[alt_e[1]][0] + float(curr_w))


                if len(tmp_cvals) != 0:
                    w_prime.update({edge: min(tmp_cvals)})
                else:
                    w_prime.update({edge: coop_dict[edge[1]][0]})

        print(f"the value of b are {set(w_prime.values())}")

        return w_prime

    def __check_edge_in_cval_play(self, edge: Tuple, play: List[Tuple]):
        """
        A helper method to check if an edge already exists in the play associated with a node in the coop dict
        :return: Return True id that exists else false
        """

        try:
            u_idx = play.index(edge[0])
            if play[u_idx + 1] == edge[1]:
                return True
        except ValueError:
            return False
        return False

    def _compute_cval_finite(self) -> Dict:
        """
        A method that pre computes all the cVals for every node in the graph and stores them in a dictionary.
        :return: A dictionary of cVal stores in dict
        """

        max_coop_val = defaultdict(lambda: '-1')
        for n in self.graph._graph.nodes():
            max_coop_val[n] = (self._compute_max_cval_from_v(n))

        return max_coop_val

    def _compute_max_cval_from_v_finite(self, node: Tuple) -> str:
        """
        A helper method to compute the cVal from a given vertex (@node) for a give graph @graph
        :param graph: The graph on which would like to compute the cVal
        :param payoff_handle: instance of the @compute_value() to compute the cVal
        :param node: The node from which we would like to compute the cVal
        :return: returns a single max value. If multiple plays have the max_value then the very first occurance is returned
        """
        tmp_copied_graph = copy.deepcopy(self.graph)
        tmp_payoff_handle = copy.deepcopy(self.payoff)
        tmp_payoff_handle.graph = tmp_copied_graph
        # construct a new graph with node as the initial vertex and compute loop_vals again
        # 1. remove the current init node of the graph
        # 2. add @node as the new init vertex
        # 3. compute the loop-vals for this new graph
        # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
        tmp_payoff_handle.remove_attribute(tmp_payoff_handle.get_init_node(), 'init')
        tmp_payoff_handle.set_init_node(node)
        tmp_payoff_handle.cycle_main()

        return tmp_payoff_handle.compute_cVal(node)

    def compute_W_prime(self):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
         alternate strategy game.
        """

        print("*****************Constructing W_prime*****************")
        coop_dict = self._compute_cval()

        w_prime: Dict[Tuple: str] = {}

        for edge in self.graph._graph.edges():

            # if the node belongs to adam, then the corresponding edge is assigned -inf
            if self.graph._graph.nodes(data='player')[edge[0]] == 'adam':
                w_prime.update({edge: -1 * math.inf})

            else:
                # a list to save all the alternate strategy cVals from a node and then selecting
                # the max of it
                tmp_cvals = []
                out_going_edge = set(self.graph._graph.out_edges(edge[0])) - set([edge])
                for alt_e in out_going_edge:
                    tmp_cvals.append(coop_dict[alt_e[1]])

                if len(tmp_cvals) != 0:
                    w_prime.update({edge: max(tmp_cvals)})
                else:
                    w_prime.update({edge: coop_dict[edge[1]]})

        print(f"the value of b are {set(w_prime.values())}")

        return w_prime

    def _compute_cval(self) -> Dict:
        """
        A method that pre computes all the cVals for every node in the graph and stores them in a dictionary.
        :return: A dictionary of cVal stores in dict
        """

        max_coop_val = defaultdict(lambda: '-1')
        for n in self.graph._graph.nodes():
            max_coop_val[n] = self._compute_max_cval_from_v(n)

        return max_coop_val

    def _compute_max_cval_from_v(self, node: Tuple) -> str:
        """
        A helper method to compute the cVal from a given vertex (@node) for a give graph @graph
        :param graph: The graph on which would like to compute the cVal
        :param payoff_handle: instance of the @compute_value() to compute the cVal
        :param node: The node from which we would like to compute the cVal
        :return: returns a single max value. If multiple plays have the max_value then the very first occurance is returned
        """
        tmp_copied_graph = copy.deepcopy(self.graph)
        tmp_payoff_handle = copy.deepcopy(self.payoff)
        tmp_payoff_handle.graph = tmp_copied_graph
        # construct a new graph with node as the initial vertex and compute loop_vals again
        # 1. remove the current init node of the graph
        # 2. add @node as the new init vertex
        # 3. compute the loop-vals for this new graph
        # tmp_payoff_handle = payoff_value(tmp_copied_graph, payoff_handle.get_payoff_func())
        tmp_payoff_handle.remove_attribute(tmp_payoff_handle.get_init_node(), 'init')
        tmp_payoff_handle.set_init_node(node)
        tmp_payoff_handle.cycle_main()

        return tmp_payoff_handle.compute_cVal(node)

    def _construct_g_b(self, g_hat: TwoPlayerGraph,
                       b: str,
                       w_prime: Dict,
                       init_node: List[Tuple],
                       accp_node: List[Tuple]) -> None:
        """

        :param g_hat:
        :param b:
        :param w_prime:
        :param init_node:
        :param accp_node:
        :return:
        """

        # each node is dict with the node name as key and 'b' as its value
        g_hat.add_states_from([((n), b) for n in self.graph._graph.nodes()])

        assert (len(init_node) == 1), f"Detected multiple init nodes in the org graph: {[n for n in init_node]}. " \
                                      f"This should not be the case"

        # assign each node a player if it hasn't been initialized yet
        for n in g_hat._graph.nodes():

            if self.graph._graph.has_node(n[0]):
                # add aps to each node in g_hat
                g_hat._graph.nodes[n]['ap'] = self.graph._graph.nodes[n[0]].get('ap')

            # assign the nodes of G_b with v1 in it at n[0] to have a 'init' attribute
            if len(n) == 2 and n[0] == init_node[0][0]:
                g_hat._graph.nodes[n]['init'] = True

            # assign the nodes of G_b with 'accepting' attribute
            for _accp_n in accp_node:
                if len(n) == 2 and n[0] == _accp_n:
                    g_hat._graph.nodes[n]['accepting'] = True

            if g_hat._graph.nodes(data='player')[n] is None:
                if self.graph._graph.nodes(data='player')[n[0]] == "adam":
                    g_hat._graph.nodes[n]['player'] = "adam"
                else:
                    g_hat._graph.nodes[n]['player'] = "eve"

        # a sample edge og g_hat: ((".","."),"."),((".","."),".") and
        # a sample edge of org_graph: (".", ""),(".", ".")
        for e in self.graph._graph.edges():
            if float(w_prime[e]) <= float(b):
                g_hat.add_edge(((e[0]), b), ((e[1]), b))

    def _construct_g_hat_nodes(self, g_hat: ProductAutomaton) -> ProductAutomaton:
        """
        A helper function that adds the nodes v0, v1 and vT that are part of g_hat graph
        :return: A updated instance of g_hat
        """

        g_hat.add_states_from(['v0', 'v1', 'vT'])

        g_hat.add_state_attribute('v0', 'player', 'adam')
        g_hat.add_state_attribute('v1', 'player', 'eve')
        g_hat.add_state_attribute('vT', 'player', 'eve')

        # add v0 as the initial node
        g_hat.add_initial_state('v0')

        # add the edges with the weights
        g_hat.add_weighted_edges_from([('v0', 'v0', '0'),
                                       ('v0', 'v1', '0'),
                                       ('vT', 'vT', str(-2 * float(self.graph.get_max_weight()) - 1))])
                                       # ('vT', 'vT', str(math.inf))])
        return g_hat

    def construct_g_hat(self,
                        w_prime: Dict[Tuple, str],
                        acc_min_edge_weight: bool = False,
                        acc_max_edge_weight: bool = False) -> TwoPlayerGraph:
        print("*****************Constructing G_hat*****************")
        # construct new graph according to the pseudocode 3

        G_hat: ProductAutomaton = graph_factory.get("ProductGraph",
                                                  graph_name="G_hat",
                                                  config_yaml="config/G_hat",
                                                  save_flag=True)
        G_hat.construct_graph()

        # build g_hat
        G_hat = self._construct_g_hat_nodes(G_hat)

        # add accepting states to g_hat
        accp_nodes = self.graph.get_accepting_states()

        # compute the range of w_prime function
        w_set = set(w_prime.values()) - {-1 * math.inf}
        org_init_nodes = self.graph.get_initial_states()

        # construct g_b
        for b in w_set - {math.inf}:
            self._construct_g_b(G_hat, b, w_prime, org_init_nodes, accp_nodes)

        # add edges between v1 of G_hat and init nodes(v1_b/ ((v1, 1), b) of graph G_b with edge weights 0
        # get init node of the org graph
        init_node_list: List[Tuple] = G_hat.get_initial_states()

        # add edge with weigh 0 from v1 to (v1,b)
        for _init_n in init_node_list:
            if isinstance(_init_n[0], tuple):
                G_hat.add_weighted_edges_from([('v1', _init_n[0], 0)])
                G_hat.remove_state_attr(_init_n[0], "init")

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
                    G_hat._graph[e[0]][e[1]][0]['weight'] = self._w_hat_b(org_edge=(e[0][0], e[1][0]),
                                                                          b_value=e[0][1])

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

    def _w_hat_b(self,
                 org_edge: Tuple[Tuple[str, str], Tuple[str, str]],
                 b_value: str) -> str:
        """
        an helper function that returns the w_hat value for a g_b graph : w_hat(e) = w(e) - b
        :param org_edge: edges of the format ("v1", "v2") or Tuple of tuples
        :param b_value:
        :return:
        """
        try:
            return str(float(self.graph._graph[org_edge[0]][org_edge[1]][0].get('weight')) - float(b_value))
        except KeyError:
            print(KeyError)
            print("The code should have never thrown this error. The error strongly indicates that the edges of the"
                  "original graph has been modified and the edge {} does not exist".format(org_edge))

    def compute_aval(self,
                     g_hat: TwoPlayerGraph,
                     w_prime: Dict,
                     optimistic: bool = False,
                     plot_all: bool = False,
                     bypass_implementation: bool = False,
                     print_reg: bool = False,
                     not_validity_check : bool = False) -> Dict[str, Dict]:
        """
        A function to compute the regret value according to algorithm 4 : Reg = -1 * Val(.,.) on g_hat
        :param g_hat: a directed multi-graph constructed using construct_g_hat()
        :param Val: a payoff function that belong to {limsup, liminf, sup, inf}
        :return: a dict consisting of the reg value and strategy for eve and adam respectively
        """
        print(f"*****************Computing regret and strategies for eve and adam*****************")
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
        str_dict, reg_flag = self._check_non_zero_regret(graph_g_hat=g_hat,
                                                         w_prime=w_prime,
                                                         str_dict=str_dict,
                                                         optimistic=optimistic,
                                                         not_validity_check=not_validity_check)
        if reg_flag:
            print("A non-zero regret exists and thus adam will play from v0 to v1 in g_hat")
        else:
            print("A non-zero regret does NOT exist and thus adam will play v0 to v0")
            if bypass_implementation:
                # # after computing all the reg value find the str with the least reg
                # min_reg_b = min(str_dict, key=lambda key: str_dict[key]['reg'])
                #
                # # return the corresponding str and reg value
                # final_str_dict.update({'reg': str_dict[min_reg_b]['reg']})
                # final_str_dict.update({'eve': str_dict[min_reg_b]['eve']})
                # final_str_dict.update({'adam': str_dict[min_reg_b]['adam']})
                # return final_str_dict
                return str_dict
            else:
                return {}

        # get the init nodes (ideally should only be one) of the org_graph
        init_node = self.graph.get_initial_states()
        max_w_org_graph: str = self.graph.get_max_weight()
        assert (len(init_node) == 1), f"Detected multiple init nodes in the org graph: {[n for n in init_node]}. " \
                                      f"This should not be the case"

        reg_threshold: str = input(f"Enter a value of threshold with the range: "
                                   f"[0, {-1 * (-2 * float(max_w_org_graph) - 1)}]: \n")

        try:
            assert 0 <= float(reg_threshold) <= -1 * (-2 * float(max_w_org_graph) - 1), "please enter a valid value " \
                                                                                           "within the above range"
        except:
            reg_threshold = input(
                f"Enter a value of threshold with the range: [0, {-1 * (-2 * float(max_w_org_graph) - 1)}]: \n")

        # update strategy for each node
        # 1. adam picks the edge with the min value
        # 2. eve picks the edge with the max value
        for b in set(w_prime.values()) - {-1 * math.inf} - {math.inf}:
            # if we haven't computed a strategy for this b value then proceed ahead
            if str_dict.get(b) is None:

                # update the str dict
                _eve_str, _adam_str = self._update_strategy_g_hat(b,
                                                                  g_hat=g_hat,
                                                                  optimistic=optimistic)
                str_dict.update({b: {'eve': _eve_str}})
                str_dict[b].update({'adam': _adam_str})

                # create an instance of ComputePayoffAndVals class
                plays_and_vals = ComputePlaysAndVals(graph=g_hat,
                                                     strategy={**_eve_str, **_adam_str},
                                                     payoff=self.payoff)

                if optimistic:
                    reg, eve_str, adam_str = plays_and_vals.get_reg_and_str_optimistic(accepting_bias=False,
                                                                                       eve_str=_eve_str,
                                                                                       adam_str=_adam_str)

                    str_dict[b]['eve'] = eve_str
                    str_dict[b]['adam'] = adam_str

                else:
                    reg = plays_and_vals.get_reg_and_str()

                str_dict[b]['reg'] = reg

            if not_validity_check:
                # check validity for every strategy in g_b(s)
                self._check_str_validity(str_dict[b], not_validity_check)

        # create a tmp dict of strategies that have reg <= reg_threshold
        __tmp_dict = {}
        for k, v in str_dict.items():
            if v['reg'] <= float(reg_threshold):
                __tmp_dict.update({k: str_dict[k]})

        if len(list(__tmp_dict.keys())) == 0:
            print(f"There does not exist any strategy within the given threshold: {reg_threshold}")
            # return {}

        # sanity check to replace all reg with -inf value with +inf
        self._str_dict_sanity_check(str_dict)

        if print_reg:
            self._print_reg_values(str_dict, plot_all=plot_all)

        if plot_all:
            return str_dict
        else:
            return self._get_least_reg_str(reg_dict=str_dict)

    def _str_dict_sanity_check(self, str_dict):

        # if plot_all:
        if len(list(str_dict.keys())) != 0:
            for k, v in str_dict.items():
                if v['reg'] == -1 * math.inf:
                    v['reg'] = math.inf
        # else:
        #     if len(list(str_dict.keys())) != 0:
        #         if str_dict['reg'] == -1 * math.inf:
        #             str_dict['reg'] = math.inf

    def _get_least_reg_str(self, reg_dict: Dict[float, Dict]) -> Dict:
        """
        A helper method that returns the least regret strategy that is also valid given a reg dictionary
         which is of the format:

        {
        b_value :
                eve: strategy
                adam: strategy
                reg: value
                valid: True/False
        }
        :param reg_dict:
        :return: the least regret dict of the form
        """
        str_dict = {}
        # sort from least to highest - ascending
        min_reg_b = sorted(reg_dict, key=lambda key: reg_dict[key]['reg'])

        for ib in min_reg_b:
            # return the corresponding str and reg value - IF THEY ARE VALID
            if reg_dict[ib].get('valid'):
                str_dict.update({'reg': reg_dict[ib]['reg']})
                str_dict.update({'valid': reg_dict[ib]['valid']})
                str_dict.update({'eve': reg_dict[ib]['eve']})
                str_dict.update({'adam': reg_dict[ib]['adam']})

                assert str_dict.get('valid') == True, \
                    "Returning an invalid strategy. This means that there exists at-least " \
                    "one transition from eve's state to the terminal state vT"
                return str_dict

    def _check_str_validity(self, str_dict: Dict, not_validity_check: bool = False) -> None:
        """
        A helper to add update the flag - true if the given strategy is valid else false
        :param str: A dictionary which is a mapping from each state to the next state
        :return: Updated the dict with the new flag - valid : True/False
        """

        # All the edges of Adam are retained in all G_b. So no node that belongs to adam has an edge to the vT -
        # the terminal state with the highest regret

        # so if we have vT in as the next node in the strategy dict then that is not a valid str.
        # NOTE: there is an exception that we manually the vT to vT loop.
        str = {**str_dict['eve'], **str_dict['adam']}
        str_dict.update({'valid': True})
        if not not_validity_check :
            for k, v in str.items():
                if 'vT' in v:
                    if k == 'vT':
                        continue
                    else:
                        str_dict['valid'] = False
                        return

    def _update_strategy_g_hat(self,
                               b: str,
                               g_hat: TwoPlayerGraph,
                               optimistic: bool = False) -> Tuple[Dict, Dict]:
        """
        A helper method that loops through each node in g_b (a part of g_hat) and updates the strategy dict with the next node.

        If optimistic is True, then we roll out from each node that belongs to eve and check which strategy is
         better when we have multiple edges from a node (of eve) with the same edge weight in g_hat.

        Returns a tuple a dictionary of eve and adam. Each dictionary is a mapping for the current node to next node(s)
        :param g_hat:
        :param
        :param optimistic:
        :return: A tuple of dictionaries for eve and adam respectively
        """

        # create dictionaries for eve and adam
        eve_str: Dict[Union[tuple, str], Union[tuple, str]] = {}
        adam_str: Dict[Union[tuple, str], Union[tuple, str]] = {}

        adam_str.update({"v0": "v1"})
        eve_str.update({'v1': ((self.graph.get_initial_states()[0][0]), b)})
        eve_str.update({'vT': 'vT'})

        for node in g_hat._graph.nodes():
            # only iterate through nodes of g_b
            if isinstance(node, tuple) and float(node[1]) == float(b):

                # if the node belongs to adam
                if g_hat._graph.nodes[node]['player'] == 'adam':
                    # get the next node and update
                    adam_str.update({node: self._get_next_node(g_hat,
                                                               node,
                                                               min,
                                                               optimistic=optimistic)})

                # if node belongs to eve
                elif g_hat._graph.nodes[node]['player'] == 'eve':
                    eve_str.update({node: self._get_next_node(g_hat,
                                                              node,
                                                              max,
                                                              optimistic=optimistic)})

                else:
                    raise warnings.warn(
                        f"The node {node} does not belong either to eve or adam. This should have "
                        f"never happened")

        return eve_str, adam_str

    def _check_non_zero_regret(self,
                               graph_g_hat: TwoPlayerGraph,
                               w_prime,
                               str_dict,
                               optimistic: bool = False,
                               not_validity_check: bool = False) -> Tuple[Dict[float, Dict], bool]:
        """
        A helper method to check if there exist non-zero regret in the game g_hat. If yes return true else False
        :param graph_g_hat: graph g_hat on which we would like to compute the regret value
        :param w_prime: the set of bs
        :return: A Tuple cflag; True if Reg > 0 else False
        """
        for b in set(w_prime.values()) - {-1 * math.inf} - {math.inf}:

            _eve_str, _adam_str = self._update_strategy_g_hat(b,
                                                              g_hat=graph_g_hat,
                                                              optimistic=optimistic)
            # update str_dict
            str_dict.update({b: {'eve': _eve_str}})
            str_dict[b].update({'adam': _adam_str})

            # create an instance of ComputePlaysAndVals class
            plays_and_vals = ComputePlaysAndVals(graph=graph_g_hat,
                                                 strategy={**_eve_str, **_adam_str},
                                                 payoff=self.payoff)

            # get regret value and the respective strategy
            if optimistic:
                reg, eve_str, adam_str = plays_and_vals.get_reg_and_str_optimistic(accepting_bias=False,
                                                                                   eve_str=_eve_str,
                                                                                   adam_str=_adam_str)
                str_dict[b]['eve'] = eve_str
                str_dict[b]['adam'] = adam_str
            else:
                reg = plays_and_vals.get_reg_and_str()

            str_dict[b]['reg'] = reg

            if not_validity_check:
                # check validity for every strategy in g_b(s)
                self._check_str_validity(str_dict[b], not_validity_check)

            if reg > 0:
                return str_dict, True

        return str_dict, False

    def _get_next_node(self, graph: TwoPlayerGraph, curr_node: Tuple, func, optimistic: bool = False) -> List[Tuple]:
        assert (
                    func == max or func == min), "Please make sure the deciding function for transitions on the game g_hat for " \
                                                 "eve and adam is either max or min"
        # NOTE: if there are multiple edges with same weight, it select the first one with the min/max value.
        #  Thus |next_node[1]| is 1.
        wt_list = {}
        for adj_edge in graph._graph.edges(curr_node):
            # get the edge weight, store it in a list and find the max/min and return the next_node
            wt_list.update({adj_edge: float(graph._graph[adj_edge[0]][adj_edge[1]][0].get('weight'))})

        if optimistic:
            threshold_value = func(wt_list.values())
            next_nodes: List[Tuple] = [k[1] for k in wt_list if wt_list[k] == threshold_value]
            return next_nodes
        else:
            threshold_value = func(wt_list.values())
            next_node = random.choice([k[1] for k in wt_list if wt_list[k] == threshold_value])
            return [next_node]

    def _get_set_of_valid_strs(self, str_dict: Dict) -> List[Dict]:
        """
        A helper method that return a list of set of valid strategies.

        This method loops throught the strategy dictionary, check if the a str_dict[b]['valid'] is True or not
         and adds it a to a list.
        :param str_dict:
        :return:
        """
        combined_strategy = []
        for str_b in str_dict.items():
            if str_b[1].get('valid'):
                combined_strategy.append({**str_b[1]['eve'], **str_b[1]['adam']})

        return combined_strategy

    def _add_strategy_flag(self,
                           g_hat: Union[TwoPlayerGraph, ProductAutomaton, Graph],
                           combined_strategy):
        """
        A helper method that adds a strategy attribute to the nodes of g_hat that belong to the strategy dict computed.

        Effect : Adds a new attribute "strategy" as False and loops over the dict and updated attribute
         to True if that node exists in the strategy dict.
        :param g_hat: The graph on which we compute the regret minimizing strategy
        :param combined_strategy: Combined dictionary of sys(eve)'s and env(adam)'s strategy
        :return:
        """

        for curr_node, next_node in combined_strategy.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    g_hat._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                g_hat._graph.edges[curr_node, next_node, 0]['strategy'] = True

    def _add_strategy_flag_only_eve(self,
                                    g_hat: Union[TwoPlayerGraph, ProductAutomaton, Graph],
                                    combined_strategy):
        """
        A helper method that adds a strategy attribute to the nodes of g_hat that belong to the strategy dict computed.

        Effect : Adds a new attribute "strategy" as False and loops over the dict and updated attribute
         to True if that node exists in the strategy dict and belongs to eve ONLY.

        :param g_hat: The graph on which we compute the regret minimizing strategy
        :param combined_strategy: Combined dictionary of sys(eve)'s and env(adam)'s strategy
        :return:
        """

        for curr_node, next_node in combined_strategy.items():
                if g_hat._graph.nodes[curr_node].get("player") == "eve":
                    if isinstance(next_node, list):
                        for n_node in next_node:
                            g_hat._graph.edges[curr_node, n_node, 0]['strategy'] = True
                    else:
                        g_hat._graph.edges[curr_node, next_node, 0]['strategy'] = True
    @deprecated
    def _from_str_b_final_play_to_str(self, g_hat: TwoPlayerGraph, combined_str: Dict):
        """
        A helper method to map back the final play least regret strategy from g_hat to the original graph
        :param g_hat:
        :param combined_str:
        :return:
        """
        # compute final loop str in g_hat
        g_hat_queue = [] # a stack that keeps tracks of nodes visited
        g_hat_str = {}

        original_str = {}

        # preprocess strategy to remove lists that have only node in them
        for k, v in combined_str.items():
            if len(v) == 1:
                combined_str[k] = v[0]

        curr_node = g_hat.get_initial_states()[0][0]
        g_hat_queue.append(curr_node)
        next_node = combined_str[curr_node]
        g_hat_str.update({curr_node: next_node})

        while next_node not in g_hat_queue:
            g_hat_queue.append(next_node)
            curr_node = next_node
            next_node = combined_str[curr_node]
            g_hat_str.update({curr_node: next_node})

        for u_node, v_node in g_hat_str.items():
            if self.graph._graph.has_edge(u_node[0], v_node[0]):
                original_str.update({u_node[0]: v_node[0]})

        return original_str

    def _from_str_b_to_str(self, g_hat: TwoPlayerGraph, combined_str: Dict):
        original_str = {}

        for k, v in combined_str.items():
            if len(v) == 1:
                combined_str[k] = v[0]

        for u_node, v_node in combined_str.items():
            if self.graph._graph.has_edge(u_node[0], v_node[0]):
                    original_str.update({u_node[0]: v_node[0]})

        return original_str

    def plot_all_str_g_hat(self,
                           g_hat: TwoPlayerGraph,
                           str_dict: Dict,
                           only_eve: bool = False,
                           plot: bool = False):
        """
        A helper method that plots all the VALID strategies computed on g_hat on g_hat. It then maps back the
         least regret strategy back to the original strategy.
        :return:
        """

        g_hat.set_edge_attribute('strategy', False)

        valid_strs = self._get_set_of_valid_strs(str_dict=str_dict)

        for str in valid_strs:
            if only_eve:
                self._add_strategy_flag_only_eve(g_hat, str)

            else:
                self._add_strategy_flag(g_hat, str)

        # get the least reg str from g_hat as we can only map that back the original graph
        least_reg_str = self._get_least_reg_str(str_dict)

        org_str = self._from_str_b_to_str(g_hat, {**least_reg_str['eve'], **least_reg_str['adam']})

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, org_str)

        else:
            self._add_strategy_flag(self.graph, org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

    def plot_str_g_hat(self,g_hat: TwoPlayerGraph,
                           str_dict: Dict,
                           only_eve: bool = False,
                           plot: bool = False):

        g_hat.set_edge_attribute('strategy', False)
        str = {**str_dict['eve'], **str_dict['adam']}

        if only_eve:
            self._add_strategy_flag_only_eve(g_hat, str)

        else:
            self._add_strategy_flag(g_hat, str)

        org_str = self._from_str_b_to_str(g_hat, str)

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, org_str)

        else:
            self._add_strategy_flag(self.graph, org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

    def _print_reg_values(self, str_dict, plot_all: bool = False):
        """
        A helper method to print the regret value for strategies in all g_b.
        :return:
        """
        # if plot_all:
        if len(list(str_dict.keys())) != 0:
            for k, v in str_dict.items():
                print(f"Reg Value for b = {k} is : {v['reg']} \n")
        # else:
        #     if len(list(str_dict.keys())) != 0:
        #         b: float = str_dict['v1'][1]
        #         print(f"Reg Value for b = {b} is : {str_dict['reg']} \n")
                # for k, v in str_dict.items():
                #     print(f"{k}: {v['reg']}")

class ComputePlaysAndVals:
    """
    A helper class consisting of methods that when given a graph and strategy dictionary,
     traverse through the strategies, compute all the possible paths.

    :param      graph:      g_hat graph
    :param      strategy    The combined strategy dictionary of eve and adam on g_hat graph
    """

    def __init__(self,
                 graph: TwoPlayerGraph,
                 strategy: Dict[Tuple, Tuple],
                 payoff: Payoff):

        self.graph = graph
        self.strategy = strategy
        self.payoff = payoff

    def _compute_all_plays(self) -> List[List[Tuple]]:
        """
        A helper method to compute all the plays possible given a strategy
        :return:
        """
        play_lst = []
        play = [self.graph.get_initial_states()[0][0]]
        for n in play:
            self._compute_all_plays_utils(n, play, play_lst)

        return play_lst

    def _compute_all_plays_utils(self,
                                 n,
                                 play,
                                 play_lst):

        if not isinstance(self.strategy[n], list):
            play.append(self.strategy[n])
            if play.count(play[-1]) >= 2:
                play_lst.append(play)
        else:
            for node in self.strategy[n]:
                path = copy.deepcopy(play)
                path.append(node)
                if path.count(path[-1]) < 2:
                    self._compute_all_plays_utils(node,
                                                  path,
                                                  play_lst)
                else:
                    play_lst.append(path)

    def find_optimal_str(self,
                         plays: List[List[Tuple]],
                         adam_str: Dict[Tuple, Tuple] = None,
                         eve_str: Dict[Tuple, Tuple] = None):
        """
        A helper method to find the most optimal strategy in a non-deterministic strategy
        :param      plays:          a list of plays(tuple)
        :param      adam_str:       A dict mapping each node that belongs to adam to the next node
        :param      eve_str:        A dict mapping each node that belongs to eve to the next node

        :return: A tuple of the min reg value, the corresponding DETERMINISTIC strategy for eve and adam
        """
        # a temp regret dict of format {play: reg_value}
        tmp_reg_dict = {}
        for play in plays:
            tmp_reg_dict.update({tuple(play): -1 * float(self._play_loop(play))})

        # after finding the strategy with the least reg value determinize the strategy
        min_reg_val = min(tmp_reg_dict.values())
        min_play = random.choice([k for k in tmp_reg_dict if tmp_reg_dict[k] == min_reg_val])

        eve_str, adam_str = self._get_str_from_play(min_play, eve_str=eve_str, adam_str=adam_str)

        return min_reg_val, eve_str, adam_str

    def find_optimal_str_acc(self,
                             plays: List[List[Tuple]],
                             adam_str: Dict[Tuple, Tuple] = None,
                             eve_str: Dict[Tuple, Tuple] = None):
        """
        A helper method to find the most optimal strategy in a non-deterministic strategy
        with bias towards accepting states

        If there exists an accepting state in a play, this function will return any ones of those plays
         (if there exists multiple)
        :param      plays:          a list of plays(tuple)
        :param      adam_str:       A dict mapping each node that belongs to adam to the next node
        :param      eve_str:        A dict mapping each node that belongs to eve to the next node
        :return: A tuple of the min reg value, the corresponding DETERMINISTIC strategy for eve and adam
        """

        def __check_acc_node(play: List[Tuple], accpeting_state) -> Tuple[bool, List[Tuple]]:
            # an inline function to help check is a play contains the accepting state or not
            # if yes then return True else False
            for state in accpeting_state:
                if state[0] in play:
                    return True, play
            return False, []

        # get accepting state(s)
        accp_nodes = self.graph.get_initial_states()
        flag = False
        for play in plays:
            flag, play = __check_acc_node(play, accp_nodes)
            if flag:
                min_reg_val = -1 * float(self._play_loop(play))
                min_play = play
                break

        if not flag:
            # a temp regret dict of format {play: reg_value}
            tmp_reg_dict = {}
            for play in plays:
                tmp_reg_dict.update({tuple(play): -1 * float(self._play_loop(play))})

            min_reg_val = min(tmp_reg_dict.values())
            min_play = random.choice([k for k in tmp_reg_dict if tmp_reg_dict[k] == min_reg_val])

        eve_str, adam_str = self._get_str_from_play(min_play, eve_str=eve_str, adam_str=adam_str)

        return min_reg_val, eve_str, adam_str

    def _get_str_from_play(self, play, adam_str, eve_str) -> Tuple[Dict, Dict]:
        """
        A helper method that maps a play (in our case a play with minimum regret) to a strategy
         dictionary for eve and adam
        :param min_play:
        :return:
        """
        # update the strategy
        for inx, node in enumerate(play):
            # only proceed up to to the second last node
            if inx < len(play) - 1:

                # if the node belongs to eve then update eve strategy
                if self.graph._graph.nodes[node]['player'] == 'adam':
                    # get the next node and update the strategy
                    adam_str[node] = play[inx + 1]

                # if the node belongs to eve then update eve strategy
                else:
                    eve_str[node] = play[inx + 1]

        return eve_str, adam_str

    def _play_loop(self, play: List[Tuple]) -> str:
        """
        helper method to compute the loop value for a given payoff function
        :param play:
        :return: The value of the loop when following the strategy @strategy
        """
        # # add nodes to this stack and as soon as a loop is found we break
        # create a tmp graph with the current node with their respective edges, compute the val and return it
        _graph = graph_factory.get('TwoPlayerGraph',
                                   graph_name="play_graph",
                                   config_yaml="",
                                   save_flag=False,
                                   pre_built=False,
                                   plot=False)

        # str_graph = nx.MultiDiGraph(name="str_graph")
        _graph.add_states_from(play)
        for i in range(0, len(play) - 1):
            _graph.add_weighted_edges_from([(play[i],
                                             play[i + 1],
                                             self.graph._graph[play[i]][play[i + 1]][0].get('weight'))])

        # add init node
        _graph.add_initial_state(play[0])

        # _graph.nodes[play[0]]['init'] = True
        # add this graph to payoff class
        # tmp_p_handle = payoff_value(str_graph, payoff_func)
        tmp_p_handle = copy.deepcopy(self.payoff)
        # tmp_p_handle = payoff_factory.get(self.payoff.payoff_func,
        #                                   graph=_graph)
        tmp_p_handle.graph = _graph
        tmp_p_handle.set_init_node(play[0])
        _loop_vals = tmp_p_handle.cycle_main()
        play_key = tuple(play)

        return _loop_vals[play_key]

    def get_reg_and_str_optimistic(self,
                                   eve_str,
                                   adam_str,
                                   accepting_bias: bool = False,) -> Tuple[float, Optional[Dict], Optional[Dict]]:
        """
        A helper method that compute all the plays for a given strategy (deterministic) and return the
         final regret value, strategy for eve and adam respectively
        :return:
        """
        plays: List[List[Tuple]] = self._compute_all_plays()

        # if deterministic strategy
        if len(plays) == 1:
            reg = -1 * float(self._play_loop(plays[0]))

        else:
            if accepting_bias:
                reg, eve_str, adam_str = self.find_optimal_str_acc(plays, eve_str=eve_str, adam_str=adam_str)
            else:
                reg, eve_str, adam_str = self.find_optimal_str(plays, eve_str=eve_str, adam_str=adam_str)

        return reg, eve_str, adam_str

    def get_reg_and_str(self) -> Tuple[float, Optional[Dict], Optional[Dict]]:
        """
        A helper method that compute all the plays for a given strategy (deterministic) and return the
         final regret value, strategy for eve and adam respectively
        :return:
        """
        plays: List[List[Tuple]] = self._compute_all_plays()
        reg = -1 * float(self._play_loop(plays[0]))

        return reg
