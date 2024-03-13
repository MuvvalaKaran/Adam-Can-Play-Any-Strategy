import abc
import warnings
import sys
import copy

from collections import defaultdict
from typing import Tuple, Optional, Dict, Union, List, Set

# import local packages
from src.graph import graph_factory
from src.graph import FiniteTransSys
from src.graph import DFAGraph
from src.graph import ProductAutomaton
from src.graph import TwoPlayerGraph

# import available str synthesis methods
from src.strategy_synthesis.regret_str_synthesis \
    import RegretMinimizationStrategySynthesis as RegMinStrSyn
from src.strategy_synthesis.adversarial_game import ReachabilityGame as ReachabilitySolver
from src.strategy_synthesis.cooperative_game import CooperativeGame
from src.strategy_synthesis.iros_solver import IrosStrategySynthesis as IrosStrSolver
from src.strategy_synthesis.value_iteration import ValueIteration, PermissiveValueIteration
from src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn


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


def compute_bounded_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                                energy_bound: int = 0,
                                debug: bool = False,
                                print_str: bool = False):

    iros_solver = IrosStrSolver(game=trans_sys, energy_bound=energy_bound, plot_modified_game=False)
    _start_state = trans_sys.get_initial_states()[0][0]
    if iros_solver.solve(debug=debug):
        print(f"There EXISTS a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")

    else:
        print(f"There DOES NOT exists a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")

    if print_str:
        iros_solver.print_map_dict()


def compute_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                        debug: bool = False,
                        permissive_strategies: bool = False,
                        print_winning_regions: bool = False,
                        print_str: bool = False,
                        plot: bool = False):

    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()

    if print_winning_regions:
        reachability_game_handle.print_winning_region()

    if print_str:
        reachability_game_handle.print_winning_strategies()

    if reachability_game_handle.is_winning():
        print("Assuming Env to be adversarial, sys CAN force a visit to the accepting states")
    else:
        print("Assuming Env to be adversarial, sys CANNOT force a visit to the accepting states")
    
    if plot:
        reachability_game_handle.plot_graph(with_strategy=True)


def play_min_max_game(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                      debug: bool = False,
                      plot: bool = False,
                      competitive: bool = True,
                      permissive_strategies: bool = False):
    
    if permissive_strategies:
        vi_handle = PermissiveValueIteration(game=trans_sys, competitive=competitive)
    else:
        vi_handle = ValueIteration(game=trans_sys, competitive=competitive)
    vi_handle.solve(debug=debug, plot=plot)
    print("******************************************************************************************************")
    print("Winning strategy exists") if vi_handle.is_winning() else print("No Winning strategy exists")
    print("******************************************************************************************************")


def play_cooperative_game(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                          debug: bool = False,
                          plot: bool = False):
    coop_handle = CooperativeGame(game=trans_sys, debug=debug, extract_strategy=True)
    coop_handle.reachability_solver()
    coop_handle.print_winning_region()
    coop_handle.print_winning_strategies()
    
    if plot:
        coop_handle.plot_graph(with_strategy=True)


def play_qual_be_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False, print_states: bool = False):
    """
     A method to compute Qualitative Best effort strategies for the system player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QualitativeBestEffortReachSyn(game=trans_sys, debug=debug)
    be_handle.compute_best_effort_strategies(plot=plot)
    be_handle.get_losing_region(print_states=print_states)
    be_handle.get_pending_region(print_states=print_states)
    be_handle.get_winning_region(print_states=print_states)


def play_quant_be_synthesis_game(trans_sys: TwoPlayerGraph, debug: bool = False, plot: bool = False, print_states: bool = False):
    """
     A method to compute Quantitative Best effort strategies for the system player
    """
    assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
    be_handle = QuantitativeBestEffortReachSyn(game=trans_sys, debug=debug)
    be_handle.compute_best_effort_strategies(plot=plot)
    be_handle.get_losing_region(print_states=print_states)
    be_handle.get_pending_region(print_states=print_states)
    be_handle.get_winning_region(print_states=print_states)
    
    # print be strategy dictionary for sanity checking
    for state, succ_states in be_handle.sys_best_effort_str.items():
        print(f"Strategy from {state} is {succ_states}")


def finite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph]):
    """
    A new regret computation method. Assumption: The weights on the graph represent costs and are hence non-negative.
    Sys player is minimizing its cumulative cost while the env player is trying to maximize the cumulative cost.

    Steps:
        1. Add an auxiliary tmp_accp state from the accepting state and the trap state. The edge weight is 0 and W_bar
           respectively. W_bar is equal to : (|V| - 1) x W where W is the max absolute weight
        2. Compute the best competitive value and best alternate value for each strategy i.e edge for sys node.
        3. Reg = competitive - cooperative(w')
        4. On this reg graph we then play competitive game i.e Sys: Min player and Env: Max player
        5. Map back these strategies to the original game.

    :param trans_sys:
    :return:
    """
    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys)

    reg_syn_handle.edge_weighted_arena_finite_reg_solver(purge_states=True,
                                                         plot=False)

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
                                         graph_name="two_player_graph4",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - a little length one
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"])

    two_player_graph.add_initial_state('s0')
    two_player_graph.add_state_attribute("s0", "player", "eve")
    two_player_graph.add_state_attribute("s1", "player", "adam")
    two_player_graph.add_state_attribute("s2", "player", "adam")
    two_player_graph.add_state_attribute("s3", "player", "eve")
    two_player_graph.add_state_attribute("s4", "player", "adam")
    two_player_graph.add_state_attribute("s5", "player", "eve")
    two_player_graph.add_state_attribute("s6", "player", "adam")
    two_player_graph.add_state_attribute("s7", "player", "adam")
    two_player_graph.add_state_attribute("s8", "player", "eve")

    two_player_graph.add_edge("s0", "s1")
    two_player_graph.add_edge("s0", "s2")
    two_player_graph.add_edge("s1", "s3")
    two_player_graph.add_edge("s2", "s5")
    two_player_graph.add_edge("s3", "s4")
    two_player_graph.add_edge("s4", "s5")
    two_player_graph.add_edge("s6", "s5")
    two_player_graph.add_edge("s3", "s6")
    two_player_graph.add_edge("s5", "s7")
    two_player_graph.add_edge("s6", "s8")
    two_player_graph.add_edge("s7", "s8")
    two_player_graph.add_edge("s7", "s0")
    two_player_graph.add_edge("s8", "s8")
    
    # reachability game
    two_player_graph.add_accepting_states_from(["s8"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


def adversarial_game_toy_example() -> TwoPlayerGraph:
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

    return two_player_graph


def construct_ltlf_dfa():
    """
    A method to construct the LTLf DFA for the given LTLf formula
    """
    
    dfa_handle = graph_factory.get('LTLfDFA',
                                    graph_name="ltlf_automaton",
                                    config_yaml="config/ltltf_automaton",
                                    save_flag=True,
                                    ltlf="!d U g",
                                    plot=True)
    sys.exit(-1)


if __name__ == "__main__":

    # define some constants
    EPSILON = 0  # 0 - the best strategy (for human too) and 1 - Completely random
    IROS_FLAG: bool = False
    ENERGY_BOUND = 30
    ALLOWED_HUMAN_INTERVENTIONS = 2

    # some constants related to computation
    finite: bool = True
    go_fast: bool = True

    # some constants that allow for appr _instance creations
    three_state_ts: bool = False
    five_state_ts: bool = False
    variant_1_paper: bool = False
    target_weighted_arena: bool = False
    two_player_arena: bool = True
    check_ltlf_dfa: bool = False

    # solver to call
    qual_BE_synthesis: bool = False
    quant_BE_synthesis: bool = True
    finite_reg_synthesis: bool = False
    infinte_reg_synthesis: bool = False
    adversarial_game: bool = False
    iros_str_synthesis: bool = False
    min_max_game: bool = False
    play_coop_game: bool = False

    # build the graph G on which we will compute the regret minimizing strategy
    if three_state_ts:
        three_state_ts_instance = ThreeStateExample(_finite=finite,
                                                    _plot_ts=True,
                                                    _plot_dfa=True,
                                                    _plot_prod=True)
        trans_sys = three_state_ts_instance.product_automaton

    elif five_state_ts:
        five_state_ts = FiveStateExample(_finite=finite,
                                         _plot_ts=False,
                                         _plot_dfa=False,
                                         _plot_prod=False)
        trans_sys = five_state_ts.product_automaton

    elif variant_1_paper:
        variant_1_instance = VariantOneGraph(_finite=finite,
                                             _plot_prod=False)
        trans_sys = variant_1_instance.product_automaton

    elif target_weighted_arena:
        twa_graph = EdgeWeightedArena(_graph_type="ewa",
                                      _plot_prod=True)
        trans_sys = twa_graph.product_automaton
    
    elif check_ltlf_dfa:
        construct_ltlf_dfa()
        sys.exit(-1)

    elif two_player_arena:
        # 4 state example
        # two_player_graph = four_state_BE_example(add_weights=True, plot=False)

        # 8 state example
        # two_player_graph = eight_state_BE_example(add_weights=True, plot=False)

        # Example 2 from Appendix
        two_player_graph = example_two_BE_example(add_weights=True, plot=False)

        # Example 3 from Appendix
        # two_player_graph = example_three_BE_example(add_weights=True, plot=True)

        # toy adversarial game graph
        # two_player_graph = adversarial_game_toy_example()

        trans_sys = two_player_graph
        # sys.exit(-1)

    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        finite_reg_minimizing_str(trans_sys)
    
    elif adversarial_game:
        compute_winning_str(trans_sys, debug=True, plot=True, print_winning_regions=True, print_str=True)
    
    elif iros_str_synthesis:
        compute_bounded_winning_str(trans_sys, energy_bound=ENERGY_BOUND, debug=False, print_str=False)
    
    elif min_max_game:
        play_min_max_game(trans_sys=trans_sys, debug=True, plot=True, permissive_strategies=True, competitive=True)
    
    elif play_coop_game:
        play_cooperative_game(trans_sys=trans_sys, debug=True, plot=True)

    elif qual_BE_synthesis:
        play_qual_be_synthesis_game(trans_sys=trans_sys, debug=True, plot=True, print_states=True)

    elif quant_BE_synthesis:
        play_quant_be_synthesis_game(trans_sys=trans_sys, debug=True, plot=True, print_states=True)

    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)