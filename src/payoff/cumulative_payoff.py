import warnings
import sys

from typing import Optional, Union

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

    3. After completion of this process, we should end with only two leaf nodes in the infinite game i.e there could
    only be two cycles, one that ends up in the accepting state and one that ends up in the trap state.
     (finite game repr)     O (start state)
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

    def __init__(self, game: Optional[TwoPlayerGraph]):
        self._game = game

    @property
    def game(self):
        return self._game

    @game.setter
    def game(self, game: TwoPlayerGraph):
        if not isinstance(game, TwoPlayerGraph):
            warnings.warn("Please enter a graph which is of type TwoPlayerGraph")

        self._game = game

    def curate_graph(self):
        """
        A method that identifies all the SCCs in a give graph and removes all the internal loops as per the class doc
        string.
        :return:
        """

        if isinstance(self.game, type(None)):
            warnings.warn("Did not find any graph. Please use the setter method to assign a graph")
            sys.exit(-1)

        # make a call to the mpg class to dump the graph data into the
        mpg_tool_handle = MpgToolBox(self.game, "org_graph")
        mpg_tool_handle.compute_SCC(go_fast=True)


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
