import warnings
import sys
import copy
import random
import networkx as nx

from typing import Optional, Union, Dict, Tuple, List, Set
from bidict import bidict

# import local packages
from src.graph import graph_factory
from src.graph import TwoPlayerGraph
from src.factory.builder import Builder
from src.mpg_tool import MpgToolBox

# import pyFAS solvers
from pyFAS.solvers import solver_factory

from helper_methods import deprecated


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
        # mpg_tool_handle = MpgToolBox(self.game, "org_graph")

        """
        If a SCC consists of more then one edge, we create a list of nodes that belong to it and then select the very
        first node. We then look at its neighbours and choose the neighbouring node that also belongs to the same
        scc(there should be atleast one such ngh node) and remove that edge.
        """
        # self._check_scc_condition(self.game, iter_var=0, nodes_pruned=[], debug=True)

        """
        A better approach : Compute SCC of the whole game ONCE - get all the SCCs with cardinality > 1. For each of
        these SCC compute the FAS and remove those edges to make acyclic while making sure that those nodes remain
        intact.
        
        """
        _local_graphs = self._get_scc()

        for _graph in _local_graphs:
            # curate each one of them until no cycle remains
            self._compute_fas(_graph)

        if debug:
            # mpg_tool_handle.compute_SCC(go_fast=True, debug=True)
            print("Yay!")

    def _get_scc(self) -> List:
        # compute the scc and return the scc graph itself
        mpg_instance = MpgToolBox(self.game, "scc_graph")
        _scc_dict = mpg_instance.compute_SCC(go_fast=True, debug=False)

        _sub_scc_graphs = []

        for scc_id, scc_nodes in _scc_dict.items():
            if len(_scc_dict[scc_id]) > 1:
                _sub_scc_graph = self._const_scc_graph(scc_nodes, self.game)
                _sub_scc_graphs.append(_sub_scc_graph)

        return _sub_scc_graphs

    @deprecated
    def _convert_weights_to_positive_costs(self, plot: bool = False):
        """
        A helper method that converts the -ve weight that represent cost to positive edge weights for a given game.
        :return:
        """

        for _e in self.game._graph.edges.data("weight"):
            _u = _e[0]
            _v = _e[1]

            _curr_weight = _e[2]
            if _curr_weight < 0:
                _new_weight: Union[int, float] = -1 * _curr_weight
            else:
                _new_weight: Union[int, float] = _curr_weight

            self.game._graph[_u][_v][0]["weight"] = _new_weight + 1

        if plot:
            self.game.plot_graph()

    @deprecated
    def _compute_edmond_algo(self):
        #convert all edges weight to postive in the game
        self._convert_weights_to_positive_costs()

        # remove the self loops at the absorbing states and add the transition to its predecessor.
        _trap_states = self.game.get_trap_states()
        _accpeting_states = self.game.get_accepting_states()

        _abs_states = _accpeting_states + _trap_states

        _exp_rm = []
        _exp_add = []
        for _s in _abs_states:
            # remove the self loop
            _exp_rm.append((_s, _s))

            for _pre_n in self.game._graph.predecessors(_s):
                if _pre_n != _s:
                    _exp_add.append((_s, _pre_n))

        self.game._graph.remove_edges_from(_exp_rm)

        for _e in _exp_add:
            self.game.add_edge(_e[0], _e[1], weight=1)


        new_graph = nx.algorithms.tree.minimum_spanning_tree(self.game._graph)
        print("Hmm")

    def _compute_fas(self, graph: TwoPlayerGraph):

        _converged = False

        # remove the self transition of absorbing states
        _trap_states = self.game.get_trap_states()
        _accpeting_states = self.game.get_accepting_states()

        _abs_states = _accpeting_states + _trap_states

        _exp_rm = []
        for _s in _abs_states:
            # remove the self loop
            _exp_rm.append((_s, _s))

        self.game._graph.remove_edges_from(_exp_rm)

        while not _converged:
            # get the solver
            self.game.plot_graph()
            _fas_solver = solver_factory.get("array_fas", graph=self.game._graph, curate_graph=True)
            _fas_solver.solve(debug=True)
            _fvs_set = _fas_solver.get_fvs_set()
            _fas_set = _fas_solver.get_fas_set()

            # now let remove en edge from each vertex in the local graph and check if the scc is acylic
            if self._remove_scc_edges(_fas_set, graph):
                _converged = True
                self.game.plot_graph()


    def _remove_scc_edges(self, fas_set, graph: TwoPlayerGraph):
        # if fas originate from human node then we remove the corresponding sys to human edge else we remove the edge

        _edges_to_be_pruned = []

        for _e in fas_set:
            _u = _e[0]
            _v = _e[1]

            if self.game.get_state_w_attribute(_u, "player") == "adam":
                _sys_node = [_pre_u for _pre_u in self.game._graph.predecessors(_u)]
                assert len(_sys_node) == 1, "Looks like there are multiple sys nodes transiting to a human node!"
                _sys_node = _sys_node[0]
                _edges_to_be_pruned.append((_sys_node, _u))

            elif self.game.get_state_w_attribute(_u, "player") == "eve":
                _edges_to_be_pruned.append((_u, _v))

            else:
                warnings.warn(f"State {_u} does not have any player associated with it.")
                sys.exit(-1)

        for edge in _edges_to_be_pruned:
            _u = edge[0]
            _v = edge[1]

            if len(list(self.game._graph.successors(_u))) == 1:
                continue
            elif len(list(self.game._graph.successors(_u))) == 0:
                raise Exception(f"Looks like we trimmed out node {_u}. The org graph should be intact")
            else:
                self.game._graph.remove_edge(_u, _v)

            # graph._graph.remove_edges_from(_edges_to_be_pruned)
            if nx.is_directed_acyclic_graph(self.game._graph):
                print("GOT A DAG, FUCK YEAH!")
                return True

        return False


    def compute_cVal(self):
        # once we are done with curating the graph, we run the cVal computation
        mpg_tool_handle = MpgToolBox(self.game, "org_graph")
        coop_dict = mpg_tool_handle.compute_cval(go_fast=True, debug=False)
        self.game.plot_graph()

    def _check_scc_condition(self,
                             local_scc_graph: TwoPlayerGraph,
                             iter_var: int,
                             nodes_pruned: list,
                             debug: bool = False):
        """
        A method that checks if all the internal loops have been broken.

        Technically this means that all the SCC will have only one element in them

        TODO: 1. An SCC will never grow after (won't include other nodes from outside the original set of nodes that
         belong to that SCC) you have removed an edge. Thus, an SCC could break into smaller SCCs with a subset of
         original nodes. So, do not re compute SCCs after removing an edge from each SCC.
         Solution: Stay in that SCC till it gets totally disbanded or breaks into SCCs that only has edges with
         zero weight
            2. Instead of going for that particular node, lets go for random edges. This would ensure that you do not
         completely remove a crucial link(block) to the goal state.

        :return:
        """
        print(f"SCC iteration : {iter_var}")
        iter_var += 1
        mpg_instance = MpgToolBox(local_scc_graph, "scc_graph")
        _scc_dict = mpg_instance.compute_SCC(go_fast=True, debug=False)

        # _converged = True
        for scc_id, nodes in _scc_dict.items():
            # lets randomly delete an edge
            if len(_scc_dict[scc_id]) > 1:
                _scc_nodes: set = set([_n for _n in _scc_dict[scc_id]])
                # _converged = False
                _local_scc_graph = self._const_scc_graph(_scc_nodes, self.game)
                # we keep on disbanding the scc until its cardinality is 1 or all its edges have 0 edge weight
                while not self._scc_converged(_scc_nodes):

                    self._remove_an_edge_in_scc(_scc_nodes, _local_scc_graph)
                    _to_prune_nodes: Set[tuple] = self._check_scc_is_total(_scc_nodes, _local_scc_graph)
                    nodes_pruned += _to_prune_nodes

                    while len(_to_prune_nodes) != 0:
                        self._remove_nodes_from(_to_prune_nodes, _local_scc_graph)
                        _scc_nodes = _scc_nodes - _to_prune_nodes
                        _to_prune_nodes = self._check_scc_is_total(_scc_nodes, _local_scc_graph)

                    # recompute the scc
                    if not self._scc_converged(_scc_nodes):
                        self._check_scc_condition(_local_scc_graph, iter_var, nodes_pruned, debug)
                    # _local_scc_graph = self._const_scc_graph(_scc_nodes, _local_scc_graph)


        # if not _converged:
        #     self._check_scc_condition(mpg_instance, iter_var, nodes_pruned)

        if debug:
            print(nodes_pruned)

    def _const_scc_graph(self, scc_nodes: set, graph: TwoPlayerGraph):
        """
        A helper method that extract the SCC given the list of scc_nodes and the graph the composes that SCC. The
        cardinality of the set of scc_nodes should be > 1.

        Returns a new instance of this graph
        :param nodes:
        :return:
        """

        assert len(scc_nodes) > 1, "Warning, trying to extract a SCC that consists of only one node."

        # create a new instance of graph
        _graph = graph_factory.get("TwoPlayerGraph",
                                   graph_name="scc_graph",
                                   config_yaml="config/scc_graph",
                                   save_flag=True,
                                   pre_built=False,
                                   from_file=False,
                                   plot=False)

        _valid_players = ["adam", "eve"]

        for _idn, _n in enumerate(scc_nodes):
            _player = self.game.get_state_w_attribute(_n, "player")
            if _player not in _valid_players:
                warnings.warn(f"Please make sure every node in graph {self.game._graph_name} has a valid player."
                              f"Currently state {_n} does not have a valid player"
                              f" associated with it.")
                sys.exit(-1)

            _graph.add_state(_n, player=_player)
            if _idn == 0:
                _graph.add_initial_state(_n)

        for _n in scc_nodes:
            _neighbours_lst = [_n for _n in graph._graph.successors(_n)]
            for _next_n in _neighbours_lst:
                if _next_n in scc_nodes:
                    _org_w: int = self.game.get_edge_weight(_n, _next_n)

                    if _graph._graph.has_edge(_n, _next_n):
                        warnings.warn(f"Looks like there exists multiple edges between : "
                                      f"{_n} ----> {_next_n}."
                                      f"Please check the original game graph G and the abstraction.")
                        sys.exit(-1)

                    _graph.add_edge(_n, _next_n, weight=_org_w)
        # _graph.plot_graph()
        return _graph

    def _scc_converged(self, scc_nodes: set) -> bool:
        """
        A helper method to check if we further need to divide up the SCC or not. This is done based on two reasons:

        1. If mean val of that SCC is 0 i.e it has finite cumulative value.  This happens only when all
         the edges in the SCC have edge weight 0

        2. If the SCC cannot be further decomposed. i.e it only consists of one node
        :return:
        """
        if len(scc_nodes) <= 1:
            return True

        # get the edges that belong to this scc
        _edges = self._get_edges_of_an_scc(scc_nodes)

        _converged = True
        for edge in _edges:
            if self.game.get_edge_attributes(*edge, "weight") != 0:
                _converged = False
                break

        return _converged

    def _get_edges_of_an_scc(self, scc_nodes: set) -> List[Tuple[tuple, tuple]]:
        """
        A method that returns a list of edges that belong to an SCC given the nodes of that SCC.

        We look at the neighbours of a node that belongs to an SCC. IF that neighbour also belongs to the same SCC then
        we add that edge to a list and return it.
        :param scc_nodes:
        :return:
        """
        _scc_edge_lst = []

        for _n in scc_nodes:
            _nghs_lst = [i for i in self.game._graph.successors(_n)]
            for _ngh_scc in scc_nodes:
                if _ngh_scc in _nghs_lst:
                    _scc_edge_lst.append((_n, _ngh_scc))

        return _scc_edge_lst

    def _remove_an_edge_in_scc(self, scc_nodes: set, scc_graph: TwoPlayerGraph):
        """
        TODO: hypothetically there could be situations where a SCC has more than one node in it but all the edges
         between those nodes could have an edge weight of 0. We need to resolve this issue.
         effect: We will end up in an infinite while loop as the SCCs will never shrink to size = 1

        :param scc_dict:
        :param node_idx_map:
        :param scc_id:
        :return:
        """
        _accp_state = self.game.get_accepting_states()[0]

        # get all the nodes that belong to this scc
        # _scc_nodes = [_n for _n in scc_dict[scc_id].keys()]
        _u = self.__choose_rand_node(nodes=scc_nodes)

        _node_ngh_lst = [i for i in scc_graph._graph.successors(_u)]
        for _ngh in _node_ngh_lst:
            if _ngh in scc_nodes:
                # we can only remove an edge that belongs to the system and has non-zero edge weight
                _v: Optional[tuple, str] = _ngh

                if self.game.get_edge_attributes(_u, _v, "weight") == 0:
                    continue

                if _v == _accp_state:
                    warnings.warn("Removing an edge to the accepting state")

                scc_graph._graph.remove_edge(_u, _v)
                self.game._graph.remove_edge(_u, _v)
                break

    def __choose_rand_node(self, nodes: set) -> Union[tuple, str]:
        """
        A helper method to choose a "random" node in a given scc. This node is the initial node of an edge.
        Thus we cannot return a human node. Also, we avoid returning the initial node as this might remove the start
        node from the graph.
        :param nodes:
        :return:
        """

        # get the initial state
        _start_state = self.game.get_initial_states()[0][0]
        # conversion to list to support choice() operation
        nodes = list(nodes)
        _n = random.choice(nodes)

        # if the list contains only two nodes, a starting node and a human node then this while loop will cause an
        # infinite loop
        if not(len(nodes) == 2 and _start_state in nodes):
            while _n == _start_state or\
                    self.game.get_state_w_attribute(_n, "player") == "adam":
                _n = random.choice(nodes)

            return _n

        """
        if you have iterated through all nodes and haven't returned anything, means all the nodes in the scc belong to
        scc, which is not a valid thing to happen. Throw an error in this case. Else if we enter the edge case of the
        list consisting of only the start state and a human node, then check that more than one edge exists from the
        start state and only then return the start state as a viable node from which we can prune an edge
        """

        # sanity check for all nodes != adam states
        _all_adam_nodes = True
        for _n in nodes:
            if self.game.get_state_w_attribute(_n, "player") != "adam":
                _all_adam_nodes = False

        if _all_adam_nodes:
            warnings.warn(f"Oops looks like an SCC in the graph only contains adam nodes. This mean that there is an"
                          f"edge between adam to adam state. The nodes are {[_n for _n in nodes]}")
            sys.exit(-1)

        # return the initial node after sanity checking
        for _n in nodes:
            if self.game.get_state_w_attribute(_n, "player") != "adam":
                assert _n == _start_state, \
                    f"Entered the edge case without the start state. The current node is {_n}"

                if len(list(self.game._graph.successors(_n))) > 1:
                    return _n

    def _remove_nodes_from(self, nodes: set, scc_graph: TwoPlayerGraph):
        """
        A helper method to remove the nodes from a given list and also remove the edges that transition to these nodes
        :param nodes:
        :return:
        """

        _edges_to_prune = []
        # for each node get the predecessor and remove that particular edge
        for _n in nodes:
            for _pre_n in scc_graph._graph.predecessors(_n):
                _edges_to_prune.append((_pre_n, _n))

        scc_graph._graph.remove_edges_from(_edges_to_prune)
        self._game._graph.remove_edges_from(_edges_to_prune)

        scc_graph._graph.remove_nodes_from(nodes)
        # self.game._graph.remove_nodes_from(nodes)

    def _check_scc_is_total(self, nodes: set, scc_graph: TwoPlayerGraph) -> Set[tuple]:
        """
        A method that returns an empty list if all the nodes of a SCC are total i.e every node in the SCC has an
        outgoing edge else a list of nodes that do not have any outgoing edges is returned
        :return:
        """
        _nodes_to_be_pruned = set()

        for _n in nodes:
            if len(list(scc_graph._graph.successors(_n))) == 0:
                _nodes_to_be_pruned.add(_n)

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
