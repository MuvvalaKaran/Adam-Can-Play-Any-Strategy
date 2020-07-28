import math
import copy
import multiprocessing
import warnings
import random
import sys

from tqdm import tqdm
from joblib import Parallel, delayed
from _collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

# import local packages
from src.graph import Graph, graph_factory
from src.graph import TwoPlayerGraph
from src.graph import ProductAutomaton
from src.payoff import Payoff
from helper_methods import deprecated

# asserts that this code is tested in linux
assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"

# needed for multi-threading w' computation
NUM_CORES = multiprocessing.cpu_count()


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

    def compute_W_prime(self, multi_thread: bool = False):
        """
        A method to compute w_prime function based on Algo 2. pseudocode.
        This function is a mapping from each edge to a real valued number - b

        b represents the best alternate value that a eve can achieve assuming Adam plays cooperatively in this
         alternate strategy game.
        """

        print("*****************Constructing W_prime*****************")
        # runner = Parallel(n_jobs=NUM_CORES, verbose=50)
        # job = delayed(self._compute_cval)
        # coop_dict = runner(job)
        print("*****************Start Parallel Processing*****************")
        coop_dict = self._compute_cval(multi_thread=multi_thread)
        print("*****************Stop Parallel Processing*****************")

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

    def _compute_cval(self, multi_thread: bool = False) -> Dict:
        """
        A method that pre computes all the cVals for every node in the graph and stores them in a dictionary.
        :return: A dictionary of cVal stores in dict
        """
        max_coop_val = defaultdict(lambda: '-1')

        if not multi_thread:
            for n in self.graph._graph.nodes():
                max_coop_val[n] = self._compute_max_cval_from_v(n)

            return max_coop_val
        else:
            runner = Parallel(n_jobs=NUM_CORES, verbose=50)
            job = delayed(self._compute_max_cval_from_v)
            results = runner(job(n) for n in self.graph._graph.nodes())

        for _n, _r in zip(self.graph._graph.nodes(), results):
            max_coop_val[_n] = _r

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
                                       # ('vT', 'vT', -100000)])
                                       ('vT', 'vT', str(-2 * abs(float(self.graph.get_max_weight())) - 1))])
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

    def _from_str_mpg_to_str(self, combined_str: Dict):
        original_str = {}
        # follow the strategy from the mpg toolbox
        node_stack = []
        curr_node = "v1"
        b_val = combined_str[curr_node][1]

        for u_node, v_node in combined_str.items():
            if self.graph._graph.has_node(u_node[0]):
                if u_node[1] == b_val and v_node[1] == b_val:
                    if self.graph._graph.has_edge(u_node[0], v_node[0]):
                            original_str.update({u_node[0]: v_node[0]})
        return original_str

    def get_controls_from_str(self, str_dict: Dict) -> List[str]:
        """
        A helper method to return a list of actions (edge labels) associated with the strategy found
        :param str_dict: The regret minimizing strategy
        :return: A sequence of labels that to be executed by the robot
        """

        start_state = self.graph.get_initial_states()[0][0]
        accepting_state = self.graph.get_accepting_states()[0]
        control_sequence = []

        curr_state = start_state
        next_state = str_dict[curr_state]
        control_sequence.append(self.graph.get_edge_attributes(curr_state, next_state, 'actions'))
        while curr_state != next_state:
            curr_state = next_state
            next_state = str_dict[curr_state]
            if next_state == accepting_state:
                break
            control_sequence.append(self.graph.get_edge_attributes(curr_state, next_state, 'actions'))

        return control_sequence

    def plot_str_from_mgp(self,
                          g_hat: TwoPlayerGraph,
                          str_dict: Dict,
                          only_eve: bool = False,
                          plot: bool = False) -> Dict:
        """
        A helper method that plots all the VALID strategies computed on g_hat on g_hat. It then maps back the
         least regret strategy back to the original strategy.
        :return:
        """

        g_hat.set_edge_attribute('strategy', False)

        if only_eve:
            self._add_strategy_flag_only_eve(g_hat, str_dict)

        else:
            self._add_strategy_flag(g_hat, str_dict)

        org_str = self._from_str_mpg_to_str(str_dict)

        if only_eve:
            self._add_strategy_flag_only_eve(self.graph, org_str)

        else:
            self._add_strategy_flag(self.graph, org_str)

        if plot:
            g_hat.plot_graph()
            self.graph.plot_graph()

        return org_str

    def _print_reg_values(self, str_dict):
        """
        A helper method to print the regret value for strategies in all g_b.
        :return:
        """
        if len(list(str_dict.keys())) != 0:
            for k, v in str_dict.items():
                print(f"Reg Value for b = {k} is : {v['reg']} \n")
