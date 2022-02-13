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
from src.strategy_synthesis import RegMinStrSyn
from src.strategy_synthesis import ReachabilitySolver
from src.strategy_synthesis import IrosStrSolver


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
                                                    config_yaml='/config/product_automaton',
                                                    trans_sys=self._trans_sys,
                                                    automaton=self._dfa,
                                                    save_flag=True,
                                                    prune=True,
                                                    debug=True,
                                                    absorbing=True,
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
                                                        config_yaml="/config/target_weighted_arena",
                                                        save_flag=True,
                                                        pre_built=True,
                                                        plot=self.plot_product)
        elif self.graph_type == "ewa":
            self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                        graph_name="edge_weighted_arena",
                                                        config_yaml="/config/edge_weighted_arena",
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
                                                    config_yaml="/config/two_player_graph",
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
                                            config_yaml="/config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="three_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=1,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        # self._dfa = graph_factory.get('DFA',
        #                               graph_name="automaton",
        #                               config_yaml="/config/automaton",
        #                               save_flag=True,
        #                               sc_ltl="F c",
        #                               use_alias=False,
        #                               plot=self.plot_dfa)
        self._dfa = graph_factory.get('PDFA',
                                      graph_name="pdfa",
                                      config_yaml="/config/PDFA_three_states_twogoals",
                                      save_flag=True,
                                      use_alias=False,
                                      plot=self.plot_dfa)


class TwoGoalsExample(GraphInstanceConstructionBase):
    """
    A class that implements the two goals transition system in the FiniteTransitionSystem class.
    """
    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False,
                 prune: bool = False,
                 plot_auto_graph: bool = False,
                 plot_trans_graph: bool = False,
                 weighting: str = 'automatonOnly',
                 complete_graph_players=['eve'],
                 integrate_accepting: bool = True,
                 multiple_accepting: bool = False):

        self._plot_auto_graph = plot_auto_graph
        self._plot_trans_graph = plot_trans_graph

        self._prune = prune
        self._weighting = weighting
        self._complete_graph_players = complete_graph_players
        self._integrate_accepting = integrate_accepting

        all_problems = True
        # all_problems = False
        if all_problems:
            # Whether to use Weight or Weights
            self._use_trans_sys_weights = True
            self._auto_config = "/config/PDFA_onegoal"
            # self._ts_config = "/config/Game_all_problems"
            # self._ts_config = "/config/Game_one_in_three_pareto_points"
            # self._ts_config = "/config/Game_three_pareto_points"
            # self._ts_config = "/config/Game_env_loops"
            self._ts_config = "/config/Game_sys_loops"

            # self._use_trans_sys_weights = False
            # self._ts_config = "/config/Game_elev_esc_stairs"
            # self._auto_config = "/config/PDFA_threegoals"
        else:
            self._use_trans_sys_weights = False
            self._auto_config = "/config/PDFA_twogoals"
            self._ts_config = "/config/Game_simple_loop"

        # self._ts_config = "/config/Game_two_goals"
        # self._ts_config = "/config/Game_simple_loop"
        # self._ts_config = "/config/Game_two_goals_self_loop"
        # self._ts_config = "/config/Game_all_problems"

        # if multiple_accepting:
        #     self._auto_config = "/config/PDFA_multiple_accepting"
        # else:
        #     # self._auto_config = "/config/PDFA_twogoals"
        #     self._auto_config = "/config/PDFA_onegoal"

        super().__init__(_finite=_finite,
                         _plot_ts=_plot_ts,
                         _plot_dfa=_plot_dfa,
                         _plot_prod=_plot_prod)

    def _build_ts(self):
        self._trans_sys = graph_factory.get(
            'TS',
            raw_trans_sys=None,
            graph_name="trans_sys",
            config_yaml=self._ts_config,
            pre_built=False,
            from_file=True,
            save_flag=True,
            debug=False,
            plot=self.plot_ts,
            human_intervention=0,
            finite=self.finite,
            plot_raw_ts=False)

    def _build_dfa(self):
        self._dfa = graph_factory.get(
            'PDFA',
            graph_name="pdfa",
            config_yaml=self._auto_config,
            save_flag=True,
            use_alias=False,
            plot=self.plot_dfa)

    def _build_product(self):
        self._product_automaton = graph_factory.get(
            'ProductGraph',
            graph_name='ProductAutomaton',
            config_yaml='/config/product_automaton',
            trans_sys=self._trans_sys,
            automaton=self._dfa,
            save_flag=True,
            prune=self._prune,
            debug=True,
            absorbing=True,
            finite=self.finite,
            plot=self.plot_product,
            plot_auto_graph=self._plot_auto_graph,
            plot_trans_graph=self._plot_trans_graph,
            weighting=self._weighting,
            complete_graph_players=self._complete_graph_players,
            integrate_accepting=self._integrate_accepting,
            use_trans_sys_weights = self._use_trans_sys_weights,
            )


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
                                            config_yaml="/config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="five_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=1,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        # self._dfa = graph_factory.get('DFA',
        #                               graph_name="automaton",
        #                               config_yaml="/config/automaton",
        #                               save_flag=True,
        #                               sc_ltl="!d U g",
        #                               use_alias=False,
        #                               plot=self.plot_dfa)
        self._dfa = graph_factory.get('PDFA',
                                      graph_name="pdfa",
                                      config_yaml="/config/PDFA_five_states_twogoals",
                                      save_flag=True,
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


def pure_game(
    trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
    cooperative: bool,
    mini_grid_instance: Optional[MinigridGraph] = None,
    epsilon: float = 0,
    max_human_interventions: int = 5,
    plot: bool = False,
    compute_reg_for_human: bool = False,
    integrate_accepting: bool = False,
    debug: bool = True,
    use_prism: bool = False):

    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()

    solver = MultiObjectiveSolver(trans_sys)
    # solver.solve(stochastic=True, adversarial=True, plot_strategies=True, bound=[5.6, 5.6])
    solver.solve(stochastic=False, adversarial=True, plot_strategies=True)

    # some constants that allow for appr _instance creations
    three_state_ts = False
    five_state_ts = False
    variant_1_paper = False
    target_weighted_arena = True

    # solver to call
    finite_reg_synthesis = True
    infinte_reg_synthesis = False
    adversarial_game = False
    iros_str_synthesis = False

    # build the graph G on which we will compute the regret minimizing strategy
    if three_state_ts:
        three_state_ts_instance = ThreeStateExample(_finite=finite,
                                                    _plot_ts=False,
                                                    _plot_dfa=False,
                                                    _plot_prod=False)
        trans_sys = three_state_ts_instance.product_automaton

def pure_adversarial_game(**kwargs):
    pure_game(cooperative=False, **kwargs)


def pure_cooperative_game(**kwargs):
    pure_game(cooperative=True, **kwargs)

    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    trans_sys = ts.product_automaton

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        finite_reg_minimizing_str(trans_sys)
    elif adversarial_game:
        compute_winning_str(trans_sys,
                            debug=True,
                            print_winning_regions=False,
                            print_str=False)
    elif iros_str_synthesis:
        compute_bounded_winning_str(trans_sys,
                                    energy_bound=ENERGY_BOUND,
                                    debug=False,
                                    print_str=False)
    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)
