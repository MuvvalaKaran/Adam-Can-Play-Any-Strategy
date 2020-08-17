import warnings
import sys
import copy
import random

from typing import Optional, Union, Dict, Tuple, List
from bidict import bidict

# import local packages
from src.graph import TwoPlayerGraph
from src.factory.builder import Builder
from src.mpg_tool import MpgToolBox


class CumulativePayoff:
    """
    This class is supposed to these following things as of right now.

    1. Run mpg toolbox to get all the SCCs. - IF a directed multigraph is strongly connected then it CONTAINS a cycle

    https://tinyurl.com/y6oru9zu

    2 Iteratively go through each SCC:
        1. remove a random non-zero edge (we might change the edge picking criteria in future).
        2. After removing an edge, check if the graph is total.
        3. For every node that does not have an outgoing edge we remove it.
        4. We remove the edges that transitioned to those nodes and continue this process until we reach a fix point.

    3. After completion of this process, we should end up with only two leaf nodes in the infinite game i.e there could
    only be two cycles, one that ends up in the accepting state and one that ends up in the trap state.
    (finite game repr)      O (start state)
                           : :
                          :   :
                         O     O
                  (acc state) (Trap state)

    Essentially we will end up with SCCs that all have a single element in it.

    From this we can infer that the SCC with 0 mean Val will have finite cumulative Val and the one with non-zero
    mean will have infinite cumulative val.

    4. Now to compute cumulative payoff from each node, we simply follow the strategy that the toolbox gives us -
    as this is the most optimal one.

        1. Cooperative game(cVal): Both players are trying to minimize their cost - essentially this is like a one player
        (MAX player).

        TODO: Verify if the optimal strategy we get using MPG toolbox aligns with the strategy from the dynamic
         programming approach for cooperative games

        2. Competitive game(aVal): Sys player is trying to minimize his/her cost while the env player is trying the
        maximize his/her cost.

    For 4.1, we follow the optimal play and compute the cumulative payoff from each node that that has 0 as its mean val
    in the mpg toolbox output.

    Using these coop values, we can construct G_hat. On this game we play a competitive game i.e 4.2. We follow steps 1,
    2, and 3. By following the pruning in 2. in G_hat the nodes, might end up in vT, trap state or accepting state. So
    if a node has a transition to vT or or trap state, then they both will have inf and hence a non-winning strategy
    will always have a reg = -inf. To avoid this,we remove the concept of vT and for every node that has no out going
    edges in G_hat we add a transition to a dummy state with a self-loop of weight zero. This way regret is finite
    for every play except the ones ending up in trap state.


    We might have to again follow the optimal play from each node (not just the plays with 0 mean value) and
    compute the cumulative payoff for that node.

    TODO: Verify, if following the optimal strategy in this game is enough or not.
    """

    # just in case, I am making a copy of the graph
    def __init__(self, game: Optional[TwoPlayerGraph]):
        self._game = copy.deepcopy(game)

    @property
    def game(self):
        return self._game

    @game.setter
    def game(self, game: TwoPlayerGraph):
        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = copy.deepcopy(game)

    def curate_graph(self, debug: bool = False):
        """
        A method that identifies all the SCCs in a give graph and removes all the internal loops as per the class doc
        string.
        :return:
        """
        print("**************************Computing SCCs*************************")

        if isinstance(self.game, type(None)):
            warnings.warn("Did not find any graph. Please use the setter method to assign a graph")
            sys.exit(-1)

        # make a call to the mpg class to dump the graph data into the
        mpg_tool_handle = MpgToolBox(self.game, "org_graph")
        # _scc_dict, node_idx_map = mpg_tool_handle.compute_SCC(go_fast=True, debug=True)

        """
        If a SCC consists of more then one edge, we create a list of nodes that belong to it and then select the very
        first node. We then look at its neighbours and choose the neighbouring node that also belongs to the same
        scc(there should be atleast one such ngh node) and remove that edge.
        """
        self._check_scc_condition(mpg_tool_handle, iter_var=0, nodes_pruned=[], debug=True)

        if debug:
            mpg_tool_handle.compute_SCC(go_fast=True, debug=True)
            print("Yay!")

    def compute_cVal(self):
        # once we are done with curating the graph, we run the cVal computation
        mpg_tool_handle = MpgToolBox(self.game, "org_graph")
        coop_dict = mpg_tool_handle.compute_cval(go_fast=True, debug=False)
        self.game.plot_graph()

    def _check_scc_condition(self,
                             mpg_instance: MpgToolBox,
                             iter_var: int,
                             nodes_pruned: list,
                             debug: bool = False):
        """
        A method that checks if all the internal loops have been broken.

        Technically this means that all the SCC will have only one element in them
        :return:
        """
        print(f"SCC iteration : {iter_var}")
        iter_var += 1
        _scc_dict, node_idx_map = mpg_instance.compute_SCC(go_fast=True, debug=False)
        _converged = True
        for scc_id, nodes in _scc_dict.items():
            # lets randomly delete an edge
            if len(_scc_dict[scc_id].keys()) > 1:
                _converged = False
                self._remove_an_edge_in_scc(_scc_dict, node_idx_map, scc_id)

                _to_prune_nodes: List[tuple] = self._check_graph_is_total()
                nodes_pruned += _to_prune_nodes
                while len(_to_prune_nodes) != 0:
                    self._remove_nodes_from(_to_prune_nodes)
                    _to_prune_nodes = self._check_graph_is_total()

        if not _converged:
            self._check_scc_condition(mpg_instance, iter_var, nodes_pruned)

        if debug:
            print(nodes_pruned)

    def _remove_an_edge_in_scc(self, scc_dict: Dict, node_idx_map: bidict, scc_id: int):
        """
        TODO: hypothetically there could be situations where a SCC has more than one node in it but all the edges
         between those nodes could have an edge weight of 0. We need to resolve this issue.
         effect: We will end up in an infinite while loop as the SCCs will never shrink to size = 1

        :param scc_dict:
        :param node_idx_map:
        :param scc_id:
        :return:
        """
        _accp_state = self.game.get_initial_states()[0][0]

        # get all the nodes that belong to this scc
        _scc_nodes = [_n for _n in scc_dict[scc_id].keys()]
        _u, _init_node = self.__choose_rand_node(node_idx_map=node_idx_map, nodes=_scc_nodes)

        _node_ngh_lst = scc_dict[scc_id][_init_node]
        for _ngh in _node_ngh_lst:
            if _ngh in _scc_nodes:
                # we can only remove an edge that belongs to the system and has non-zero edge weight
                _v: Optional[tuple, str] = node_idx_map.inverse[_ngh]

                if self.game.get_edge_attributes(_u, _v, "weight") == 0:
                    continue

                if _v == _accp_state:
                    warnings.warn("Removing an edge to the accepting state")

                self.game._graph.remove_edge(_u, _v)
                break

    def __choose_rand_node(self, node_idx_map: bidict, nodes: list) -> Tuple[tuple, int]:
        """
        A helper method to choose a "random" node in a given scc. This node is the initial node of an edge.
        Thus we cannot return a human node. Also, we avoid returning the initial node as this might remove the start
        node from the graph.
        :param nodes:
        :return:
        """

        # get the initial state
        _start_state = self.game.get_initial_states()[0][0]

        for _n in nodes:
            if node_idx_map.inverse[_n] != _start_state and\
                    self.game.get_state_w_attribute(node_idx_map.inverse[_n], "player") != "adam":
                return node_idx_map.inverse[_n], _n
        """
        if you have iterated through all nodes and haven't returned anything, means all the nodes in the scc belong to
        scc, which is not a valid thing to happen. Throw an error in this case. Else if we enter the edge case of the
        list consisting of only the start state and a human node, then check that more than one edge exists from the
        start state and only then return the start state as a viable node from which we can prune an edge
        """

        # sanity check for all nodes != adam states
        _all_adam_nodes = True
        for _n in nodes:
            if self.game.get_state_w_attribute(node_idx_map.inverse[_n], "player") != "adam":
                _all_adam_nodes = False

        if _all_adam_nodes:
            warnings.warn(f"Oops looks like an SCC in the graph only contains adam nodes. This mean that there is an"
                          f"edge between adam to adam state. The nodes are {[_n for _n in nodes]}")
            sys.exit(-1)

        # return the initial node after sanity checking
        for _n in nodes:
            if self.game.get_state_w_attribute(node_idx_map.inverse[_n], "player") != "adam":
                assert node_idx_map.inverse[_n] == _start_state, \
                    f"Entered the edge case without the start state. The current node is {node_idx_map}"

                if len(list(self.game._graph.successors(node_idx_map.inverse[_n]))) > 1:
                    return node_idx_map.inverse[_n], _n

    def _remove_nodes_from(self, nodes: list):
        """
        A helper method to remove the nodes from a given list and also remove the edges that transition to these nodes
        :param nodes:
        :return:
        """

        _edges_to_prune = []
        # for each node get the predecessor and remove that particular edge
        for _n in nodes:
            for _pre_n in self.game._graph.predecessors(_n):
                _edges_to_prune.append((_pre_n, _n))

        self.game._graph.remove_edges_from(_edges_to_prune)

        self.game._graph.remove_nodes_from(nodes)

    def _check_graph_is_total(self) -> list:
        """
        A method that returns an empty list if the graph is total i.e every node in the graph has an outgoing edge else
        a list of nodes that do not have any outgoing edges
        :return:
        """
        _nodes_to_be_pruned = []

        for _n in self.game._graph.nodes():
            if len(list(self.game._graph.successors(_n))) == 0:
                _nodes_to_be_pruned.append(_n)

        return _nodes_to_be_pruned


class CumulativePayoffBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self, graph: Optional[TwoPlayerGraph], payoff_string: str):
        """
        A method that returns a concrete instance of Cumulative class
        :param graph:
        :return:
        """

        self._instance = CumulativePayoff(game=graph)

        return self._instance
