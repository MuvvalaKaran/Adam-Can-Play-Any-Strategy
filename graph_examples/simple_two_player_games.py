import abc
import warnings
import sys

from typing import Optional

# import local packages
from src.graph import graph_factory
from src.graph import FiniteTransSys
from src.graph import DFAGraph
from src.graph import ProductAutomaton
from src.graph import TwoPlayerGraph


class GraphInstanceConstructionBase(abc.ABC):
    """
    An abstract class acting as interface to build a graph which is the input the regret minimizing strategy class.

    finite: flag indicating that we are using finite/cumulative payoff which alters the transition system
    and product automaton graph construction at the fundamental level. The flag manipulates the weights associated with
    the absorbing states(if any) in raw transition system and the absorbing states in product automaton.
    """
    human_intervention: int = 2
    human_intervention_cost: int = 0
    human_non_intervention_cost: int = 0

    def __init__(self,
                 _finite: bool,
                 _plot_ts: bool,
                 _plot_dfa: bool,
                 _plot_prod: bool):
        self.finite = _finite
        self.plot_ts = _plot_ts
        self.plot_dfa = _plot_dfa
        self.plot_product = _plot_prod

        self._trans_sys: Optional[FiniteTransSys] = None
        self._dfa: Optional[DFAGraph] = None
        self._product_automaton: Optional[ProductAutomaton] = None

        self._build_ts()
        self._build_dfa()
        self._build_product()

    @abc.abstractmethod
    def _build_ts(self):
        pass

    @abc.abstractmethod
    def _build_dfa(self):
        pass

    def _build_product(self):
        self._product_automaton = graph_factory.get('ProductGraph',
                                                    graph_name='product_automaton',
                                                    config_yaml='config/product_automaton',
                                                    trans_sys=self._trans_sys,
                                                    automaton=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
                                                    absorbing=True,
                                                    view=False,
                                                    finite=self.finite,
                                                    plot=self.plot_product)

    @property
    def product_automaton(self):
        return self._product_automaton


class EdgeWeightedArena(GraphInstanceConstructionBase):
    """
    A class that constructs concrete instance of Edge Weighted arena as per Filliot's paper.
    """
    def __init__(self,
                 _graph_type: str,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        self.graph_type = _graph_type
        self._initialize_graph_type()
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _initialize_graph_type(self):
        valid_graph_type = ['ewa', 'twa']

        if self.graph_type not in valid_graph_type:
            warnings.warn(f"The current graph type {self.graph_type} doe not bleong to the set of valid graph options"
                          f"{valid_graph_type}")
            sys.exit(-1)

    def _build_dfa(self):
        pass

    def _build_ts(self):
        pass

    def _build_product(self):
        if self.graph_type == "twa":
            self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                        graph_name="target_weighted_arena",
                                                        config_yaml="config/target_weighted_arena",
                                                        from_file=True,
                                                        save_flag=True,
                                                        plot=self.plot_product)
        elif self.graph_type == "ewa":
            self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                        graph_name="edge_weighted_arena",
                                                        config_yaml="config/edge_weighted_arena",
                                                        from_file=True,
                                                        save_flag=True,
                                                        plot=self.plot_product)
        else:
            warnings.warn("PLease enter a valid graph type")


class VariantOneGraph(GraphInstanceConstructionBase):
    """
    A class that constructs a concrete instance of the TwoPlayerGraph(G) based on the example on Pg 2. of the paper.

    We then compute a regret minimizing strategy on G.
    """

    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _build_dfa(self):
        pass

    def _build_ts(self):
        pass

    def _build_product(self):
        self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                    graph_name="two_player_graph",
                                                    config_yaml="config/two_player_graph",
                                                    from_file=True,
                                                    save_flag=True,
                                                    plot=self.plot_product)


class ThreeStateExample(GraphInstanceConstructionBase):
    """
    A class that implements the built-in three state raw transition system in the FiniteTransitionSystem class. We then
    build a concrete instance of a transition system by augmenting the graph with human/env nodes. Given a fixed
    syntactically co-safe LTL formula(!b U c) we construct the Product Automation (G) on which we then compute a regret
    minimizing strategy.
    """
    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _build_ts(self):
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=None,
                                            graph_name="trans_sys",
                                            config_yaml="config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="three_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=1,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="config/automaton",
                                      save_flag=True,
                                      sc_ltl="F c",
                                      use_alias=False,
                                      view=False,
                                      plot=self.plot_dfa)


class FiveStateExample(GraphInstanceConstructionBase):
    """
    A class that implements the built-in five state raw transition system in the FiniteTransitionSystem class. We then
    build a concrete instance of a transition system by augmenting the graph with human/env nodes. Given a fixed
    syntactically co-safe LTL formula(!d U g) we construct the Product Automation (G) on which we then compute a regret
    minimizing strategy.
    """
    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _build_ts(self):
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=None,
                                            graph_name="trans_sys",
                                            config_yaml="config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="five_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=1,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="config/automaton",
                                      save_flag=True,
                                      sc_ltl="!d U g",
                                      use_alias=False,
                                      plot=self.plot_dfa)



   
def four_state_BE_example(add_weights: bool = False, plot: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 4 state toy exmaple from our discussion to test Strategy synthesis. Fig. 1. in our paper
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph1",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - Simple one
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4"])
    two_player_graph.add_initial_state('s0')
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "eve")
    two_player_graph.add_state_attribute("s3", "player", "adam")
    two_player_graph.add_state_attribute("s4", "player", "eve")

    two_player_graph.add_edge("s0", "s1", weight=1)
    two_player_graph.add_edge("s0", "s3", weight=1)
    two_player_graph.add_edge("s1", "s0", weight=1)
    two_player_graph.add_edge("s1", "s2", weight=1)
    two_player_graph.add_edge("s2", "s2", weight=1)
    two_player_graph.add_edge("s3", "s4", weight=1)
    two_player_graph.add_edge("s3", "s2", weight=1)
    two_player_graph.add_edge("s4", "s4", weight=1)

    
    two_player_graph.add_accepting_states_from(["s4"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0
    
    if plot:
        two_player_graph.plot_graph()
    
    return two_player_graph


def eight_state_BE_example(add_weights: bool = False, plot: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 8 state toy exmaple form our discussion to test Sstrategy synthesis
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph2",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - a little length one
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])

    # temp add a cycle at the init state
    two_player_graph.add_states_from(["s8"])
    two_player_graph.add_state_attribute("s8", "player", "adam")
    two_player_graph.add_edge("s0", "s8")
    two_player_graph.add_edge("s8", "s0")

    two_player_graph.add_initial_state('s0')
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "adam")
    two_player_graph.add_state_attribute("s3", "player", "eve")
    two_player_graph.add_state_attribute("s4", "player", "eve")
    two_player_graph.add_state_attribute("s5", "player", "adam")
    two_player_graph.add_state_attribute("s6", "player", "eve")
    two_player_graph.add_state_attribute("s7", "player", "eve")

    two_player_graph.add_edge("s0", "s1")
    two_player_graph.add_edge("s0", "s2")
    two_player_graph.add_edge("s1", "s0")
    two_player_graph.add_edge("s1", "s4")
    two_player_graph.add_edge("s2", "s3")
    two_player_graph.add_edge("s3", "s3")
    two_player_graph.add_edge("s2", "s7")
    two_player_graph.add_edge("s4", "s5")
    two_player_graph.add_edge("s5", "s4")
    two_player_graph.add_edge("s5", "s6")
    two_player_graph.add_edge("s5", "s7")
    two_player_graph.add_edge("s6", "s6")
    two_player_graph.add_edge("s7", "s7")

    # adding dangling env and sys state for testing compute_cooperative_winning_strategy synthesis for env and sys player
    two_player_graph.add_states_from(["s9", "s10"])
    two_player_graph.add_state_attribute("s9", "player", "adam")
    two_player_graph.add_state_attribute("s10", "player", "eve") 
    two_player_graph.add_edge("s9", "s10")
    two_player_graph.add_edge("s10", "s9")

    two_player_graph.add_states_from(["s11", "s12"])
    two_player_graph.add_state_attribute("s11", "player", "eve")
    two_player_graph.add_state_attribute("s12", "player", "eve")
    two_player_graph.add_edge("s8", "s11")
    two_player_graph.add_edge("s11", "s11")
    two_player_graph.add_edge("s1", "s12")
    two_player_graph.add_edge("s12", "s12")

    
    # safety game
    # two_player_graph.add_accepting_states_from(["s0", "s4", "s7"])
    
    # reachability game
    two_player_graph.add_accepting_states_from(["s7"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0
    
    ## Testing -manually changing the edge wegit s0 -> s2
    # two_player_graph.add_weighted_edges_from(["s0", "s2", 2])
    two_player_graph._graph["s0"]["s2"][0]["weight"] =  1

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph



def example_two_BE_example(add_weights: bool = False, plot: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 11 state exmaple form our discussion to test Sstrategy synthesis (Example 2)
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph3",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - a little length one
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"])

    two_player_graph.add_initial_state('s0')
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "eve")
    two_player_graph.add_state_attribute("s3", "player", "adam")
    two_player_graph.add_state_attribute("s4", "player", "eve")
    two_player_graph.add_state_attribute("s5", "player", "adam")
    two_player_graph.add_state_attribute("s6", "player", "eve")
    two_player_graph.add_state_attribute("s7", "player", "adam")
    two_player_graph.add_state_attribute("s8", "player", "eve")
    two_player_graph.add_state_attribute("s9", "player", "adam")
    two_player_graph.add_state_attribute("s10", "player", "eve")

    two_player_graph.add_edge("s0", "s1")
    two_player_graph.add_edge("s1", "s0")
    two_player_graph.add_edge("s1", "s2")
    two_player_graph.add_edge("s1", "s6")
    two_player_graph.add_edge("s2", "s3")
    two_player_graph.add_edge("s3", "s2")
    two_player_graph.add_edge("s3", "s4")
    two_player_graph.add_edge("s4", "s5")
    two_player_graph.add_edge("s5", "s4")
    two_player_graph.add_edge("s5", "s6")
    two_player_graph.add_edge("s6", "s7")
    two_player_graph.add_edge("s7", "s8")
    two_player_graph.add_edge("s8", "s9")
    two_player_graph.add_edge("s9", "s10")
    two_player_graph.add_edge("s10", "s10")
    
    # reachability game
    two_player_graph.add_accepting_states_from(["s10"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


def example_three_BE_example(add_weights: bool = False, plot: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 8 state exmaple from our discussion to test Strategy synthesis (Example 3)
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph5",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - a little length one
    two_player_graph.add_states_from(["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"])

    two_player_graph.add_initial_state('s1')
    two_player_graph.add_state_attribute("s1", "player", "eve")
    two_player_graph.add_state_attribute("s2", "player", "adam")
    two_player_graph.add_state_attribute("s3", "player", "adam")
    two_player_graph.add_state_attribute("s4", "player", "eve")
    two_player_graph.add_state_attribute("s5", "player", "adam")
    two_player_graph.add_state_attribute("s6", "player", "eve")
    two_player_graph.add_state_attribute("s7", "player", "adam")
    two_player_graph.add_state_attribute("s8", "player", "adam")
    two_player_graph.add_state_attribute("s9", "player", "eve")

    two_player_graph.add_edge("s1", "s2")
    two_player_graph.add_edge("s1", "s3")
    two_player_graph.add_edge("s2", "s4")
    two_player_graph.add_edge("s3", "s6")
    two_player_graph.add_edge("s4", "s5")
    two_player_graph.add_edge("s5", "s6")
    two_player_graph.add_edge("s7", "s6")
    two_player_graph.add_edge("s4", "s7")
    two_player_graph.add_edge("s6", "s8")
    two_player_graph.add_edge("s7", "s9")
    two_player_graph.add_edge("s8", "s9")
    two_player_graph.add_edge("s8", "s1")
    two_player_graph.add_edge("s8", "s4")
    two_player_graph.add_edge("s9", "s9")
    
    # reachability game
    two_player_graph.add_accepting_states_from(["s9"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


def adversarial_game_toy_example(plot: bool = False) -> TwoPlayerGraph:
    """
    The example from the Adversarial Game script. 
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam)
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "adam")
    two_player_graph.add_state_attribute("s3", "player", "adam")
    two_player_graph.add_state_attribute("s4", "player", "eve")
    two_player_graph.add_state_attribute("s5", "player", "adam")
    two_player_graph.add_state_attribute("s6", "player", "eve")
    two_player_graph.add_state_attribute("s7", "player", "adam")

    two_player_graph.add_edge("s0", "s1")
    two_player_graph.add_edge("s0", "s3")
    two_player_graph.add_edge("s1", "s0")
    two_player_graph.add_edge("s1", "s2")
    two_player_graph.add_edge("s1", "s4")
    two_player_graph.add_edge("s2", "s2")
    two_player_graph.add_edge("s2", "s4")
    two_player_graph.add_edge("s3", "s4")
    two_player_graph.add_edge("s3", "s0")
    two_player_graph.add_edge("s3", "s5")
    two_player_graph.add_edge("s4", "s3")
    two_player_graph.add_edge("s4", "s1")
    two_player_graph.add_edge("s5", "s3")
    two_player_graph.add_edge("s5", "s6")
    two_player_graph.add_edge("s6", "s6")
    two_player_graph.add_edge("s6", "s7")
    two_player_graph.add_edge("s7", "s3")
    two_player_graph.add_edge("s7", "s0")

    two_player_graph.add_accepting_states_from(["s3", "s4"])

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


