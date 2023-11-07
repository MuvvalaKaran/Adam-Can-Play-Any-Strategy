import abc
import warnings
import sys

from typing import Tuple, Optional, Dict, Union

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
from src.strategy_synthesis.iros_solver import IrosStrategySynthesis as IrosStrSolver
from src.strategy_synthesis.value_iteration import ValueIteration


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
                                                        pre_built=True,
                                                        plot=self.plot_product)
        elif self.graph_type == "ewa":
            self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                        graph_name="edge_weighted_arena",
                                                        config_yaml="config/edge_weighted_arena",
                                                        from_file=True,
                                                        save_flag=True,
                                                        pre_built=True,
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
                                                    pre_built=True,
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
                        print_winning_regions: bool = False,
                        print_str: bool = False):

    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()
    _sys_str_dict = reachability_game_handle.sys_str
    _env_str_dict = reachability_game_handle.env_str

    if print_winning_regions:
        reachability_game_handle.print_winning_region()

    if print_str:
        reachability_game_handle.print_winning_strategies()

    if reachability_game_handle.is_winning():
        print("Assuming Env to be adversarial, sys CAN force a visit to the accepting states")
    else:
        print("Assuming Env to be adversarial, sys CANNOT force a visit to the accepting states")


def play_min_max_game(trans_sys: Union[FiniteTransSys, TwoPlayerGraph],
                      debug: bool = False,
                      plot: bool = False):
    
    vi_handle = ValueIteration(game=trans_sys, competitive=True)
    vi_handle.solve(debug=debug, plot=plot)



def finite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph]):
    """
    A new regret computation method. Assumption: The weights on the graph represent costs and are hence non-negative.
    Sys player is trying to minimize its cumulative cost while the env player is trying to maximize the cumulative cost.

    Steps:

        1. Add an auxiliary tmp_accp state from the accepting state and the trap state. The edge weight is 0 and W_bar
           respectively. W_bar is equak to : (|V| - 1) x W where W is the max absolute weight
        2. Compute the best competitive value and best alternate value for each strategy i.e edge for sys node.
        3. Reg = competitive - cooperative(w')
        4. On this reg graph we then play competitive game i.e Sys: Min player and Env: Max player
        5. Map back these strategies to the original game.

    :param trans_sys:
    :param mini_grid_instance:
    :param epsilon:
    :param max_human_interventions:
    :param plot:
    :return:
    """

    # payoff = payoff_factory.get("cumulative", graph=trans_sys)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys)

    # if mini_grid_instance:
    #     reg_syn_handle.add_common_accepting_state(plot=False)

    # reg_syn_handle.target_weighted_arena_finite_reg_solver(twa_graph=trans_sys,
    #                                                        debug=False,
    #                                                        purge_states=True,
    #                                                        plot_w_vals=True,
    #                                                        plot=plot,
    #                                                        plot_only_eve=False)

    reg_syn_handle.edge_weighted_arena_finite_reg_solver(purge_states=True,
                                                         plot=False)

    # reg_syn_handle.finite_reg_solver_1(minigrid_instance=mini_grid_instance,
    #                                    plot=plot,
    #                                    plot_only_eve=False,
    #                                    simulate_minigrid=bool(mini_grid_instance),
    #                                    epsilon=epsilon,
    #                                    max_human_interventions=max_human_interventions,
    #                                    compute_reg_for_human=compute_reg_for_human)

    # reg_syn_handle.finite_reg_solver_2(minigrid_instance=mini_grid_instance,
    #                                    plot=plot,
    #                                    plot_only_eve=False,
    #                                    simulate_minigrid=bool(mini_grid_instance),
    #                                    epsilon=epsilon,
    #                                    max_human_interventions=max_human_interventions,
    #                                    compute_reg_for_human=compute_reg_for_human)


def four_state_BE_example(add_weights: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 4 state toy exmaple form our discussion to test Sstrategy synthesis
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         pre_built=False,
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
    return two_player_graph


def eight_state_BE_example(add_weights: bool = False) -> TwoPlayerGraph:
    """
    A method where I manually create the 4 state toy exmaple form our discussion to test Sstrategy synthesis
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         pre_built=False,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam) - a little length one
    two_player_graph.add_states_from(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])

    # temp add a cycle at the init state
    # two_player_graph.add_states_from(["s8"])
    # two_player_graph.add_state_attribute("s8", "player", "adam")
    # two_player_graph.add_edge("s0", "s8")
    # two_player_graph.add_edge("s8", "s0")

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
    
    # safety game
    # two_player_graph.add_accepting_states_from(["s0", "s4", "s7"])
    
    # reachability game
    two_player_graph.add_accepting_states_from(["s7"])

    if add_weights:
        for _s in two_player_graph._graph.nodes():
            for _e in two_player_graph._graph.out_edges(_s):
                two_player_graph._graph[_e[0]][_e[1]][0]["weight"] = 1 if two_player_graph._graph.nodes(data='player')[_s] == 'eve' else 0
    
    return two_player_graph


def adversarial_game_toy_example() -> TwoPlayerGraph:
    """
    The two example found from the Adversarial Game script. 
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="/config/two_player_graph",
                                         save_flag=True,
                                         pre_built=False,
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


if __name__ == "__main__":

    # define some constants
    EPSILON = 0  # 0 - the best strategy (for human too) and 1 - Completely random
    IROS_FLAG = False
    ENERGY_BOUND = 30
    ALLOWED_HUMAN_INTERVENTIONS = 2

    # some constants related to computation
    finite = True
    go_fast = True

    # some constants that allow for appr _instance creations
    three_state_ts = False
    five_state_ts = False
    variant_1_paper = False
    target_weighted_arena = False
    two_player_arena = True

    # solver to call
    BE_synthesis = False
    finite_reg_synthesis = False
    infinte_reg_synthesis = False
    adversarial_game = False
    iros_str_synthesis = False
    min_max_game = True

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
    
    elif two_player_arena:
        # 4 state example
        # two_player_graph = four_state_BE_example(add_weights=True)

        # 6 state example
        two_player_graph = eight_state_BE_example(add_weights=True)

        # toy adversarial game graph
        # two_player_graph = adversarial_game_toy_example()

        trans_sys = two_player_graph

    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        finite_reg_minimizing_str(trans_sys)
    elif adversarial_game:
        compute_winning_str(trans_sys,
                            debug=True,
                            print_winning_regions=True,
                            print_str=True)
    elif iros_str_synthesis:
        compute_bounded_winning_str(trans_sys,
                                    energy_bound=ENERGY_BOUND,
                                    debug=False,
                                    print_str=False)
    elif min_max_game:
        play_min_max_game(trans_sys=trans_sys, debug=True, plot=True)

    elif BE_synthesis:
        assert isinstance(trans_sys, TwoPlayerGraph), "Make sure the graph is an instance of TwoPlayerGraph class for Best effort experimental code."
        reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=True)
        reachability_game_handle.reachability_solver()
        # reachability_game_handle.maximally_permissive_reachability_solver()
        reachability_game_handle.plot_graph(with_strategy=True)
        reachability_game_handle.print_winning_region()
        reachability_game_handle.print_winning_strategies()

    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)