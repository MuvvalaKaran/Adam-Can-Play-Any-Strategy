import os
import gym
import enum
import abc
import warnings
import sys
import copy
import argparse

import numpy as np
from typing import Tuple, Optional, Dict, Union

# import wombats packages
from wombats.systems import StaticMinigridTSWrapper
from wombats.automaton import active_automata
from wombats.automaton import MinigridTransitionSystem

# import local packages
from src.graph import graph_factory
from src.payoff import payoff_factory
from src.graph import MiniGrid
from src.graph import FiniteTransSys
from src.graph import DFAGraph
from src.graph import ProductAutomaton
from src.graph import TwoPlayerGraph

# import available str synthesis methods
from src.strategy_synthesis import RegMinStrSyn
from src.strategy_synthesis import ReachabilitySolver
from src.strategy_synthesis import IrosStrSolver
from src.strategy_synthesis import ValueIteration
from src.strategy_synthesis import MultiObjectiveSolver
from src.prism import PrismInterfaceForTwoPlayerGame


from src.mpg_tool import MpgToolBox

# assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"

# directory where we will be storing all the configuration files related to graphs
DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(DIR, 'config')


"""
Note: When you create a NxN world, two units of width and height are consumed for drawing the boundary.
So a 4x4 world will be a 2x2 env and 5x5 will be a 3x3 env respectively.
"""


class MiniGridEmptyEnv(enum.Enum):
    env_4 = 'MiniGrid-Empty-4x4-v0'
    env_5 = 'MiniGrid-Empty-5x5-v0'
    env_6 = 'MiniGrid-Empty-6x6-v0'
    env_8 = 'MiniGrid-Empty-8x8-v0'
    env_16 = 'MiniGrid-Empty-16x16-v0'
    renv_3 = 'MiniGrid-Empty-Random-3x3-v0'
    renv_4 = 'MiniGrid-Empty-Random-4x4-v0'
    renv_5 = 'MiniGrid-Empty-Random-5x5-v0'
    renv_6 = 'MiniGrid-Empty-Random-6x6-v0'


class MiniGridLavaEnv(enum.Enum):
    env_1 = 'MiniGrid-DistShift1-v0'
    env_2 = 'MiniGrid-DistShift2-v0'
    env_3 = 'MiniGrid-LavaGapS5-v0'
    env_4 = 'MiniGrid-LavaGapS6-v0'
    env_5 = 'MiniGrid-LavaGapS7-v0'
    env_6 = 'MiniGrid-Lava_NoEntry-v0'
    env_7 = 'MiniGrid-Lava_SmallEntry-v0'


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


class MinigridGraph(GraphInstanceConstructionBase):
    """
    A concrete implementation of an instance of FiniteTransitionSystem from an env in gym-minigrid. Given an Env we
    build a "raw transition system" that only includes system nodes. We then add human/env nodes by using an instance of
    FiniteTransitionSystem(TS) and building the concrete instance. Given a fixed syntactically co-safe LTL formula, we
    compose the TS and the DFA to get an instance of the product automaton(G). We compute a regret Minimizing strategy
    on this Product Graph G.

    wombat_minigrid_TS: An concrete instance of MiniGridTransitionSystem from wombats tool given an env.
    """

    def __init__(self,
                 _finite: bool = False,
                 _iros_ts: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False,
                 _plot_minigrid: bool = False):
        self._wombats_minigrid_TS: Optional[MinigridTransitionSystem] = None
        self._plot_minigrid = _plot_minigrid
        self.get_iros_ts = _iros_ts
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def __get_TS_from_wombats(self) -> Tuple[MiniGrid, MinigridTransitionSystem]:
        # ENV_ID = 'MiniGrid-LavaComparison_noDryingOff-v0'
        # ENV_ID = 'MiniGrid-AlternateLavaComparison_AllCorridorsOpen-v0'
        # ENV_ID = 'MiniGrid-DistShift1-v0'
        # ENV_ID = 'MiniGrid-LavaGapS5-v0'
        # ENV_ID = 'MiniGrid-Empty-5x5-v0'
        # ENV_ID = MiniGridEmptyEnv.env_6.value
        # ENV_ID = MiniGridLavaEnv.env_6.value
        ENV_ID = 'MiniGrid-Lava_Multiple_Goals_SmallEntry-v0'
        # ENV_ID = 'MiniGrid-MyDistShift-v0'

        env = gym.make(ENV_ID)
        env = StaticMinigridTSWrapper(env, actions_type='simple_static')
        env.render()

        wombats_minigrid_TS = active_automata.get(automaton_type='TS',
                                                  graph_data=env,
                                                  graph_data_format='minigrid')

        # file to dump the TS corresponding to the gym env
        file_name = ENV_ID + '_TS'
        abs_file_path = os.path.join(CONFIG_DIR, file_name + ".yaml")
        wombats_minigrid_TS.to_yaml_file(abs_file_path)

        regret_minigrid_TS = graph_factory.get('MiniGrid',
                                               graph_name="minigrid_TS",
                                               config_yaml=f"/config/{file_name}",
                                               save_flag=True,
                                               plot=False)

        return regret_minigrid_TS, wombats_minigrid_TS

    def execute_str(self, _controls):
        self._wombats_minigrid_TS.run(_controls, record_video=True, show_steps=True)

    def _build_ts(self):
        raw_trans_sys, self._wombats_minigrid_TS = self.__get_TS_from_wombats()

        self._trans_sys = graph_factory.get('MiniGrid',
                                            raw_minigrid_ts=raw_trans_sys,
                                            get_iros_ts=self.get_iros_ts,
                                            graph_name=raw_trans_sys._graph_name,
                                            config_yaml=raw_trans_sys._config_yaml,
                                            human_interventions=self.human_intervention,
                                            human_intervention_cost=self.human_intervention_cost,
                                            human_non_intervention_cost=self.human_non_intervention_cost,
                                            save_flag=True,
                                            plot_raw_minigrid=self._plot_minigrid,
                                            plot=self.plot_ts)

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="/config/automaton",
                                      save_flag=True,
                                      # sc_ltl="!(lava_red_open) U(carpet_yellow_open) &(!(lava_red_open) U (water_blue_open))",
                                      # sc_ltl="!(lava_red_open) U (water_blue_open)",
                                      # sc_ltl="!(lava_red_open) U (goal_green_open)",
                                      sc_ltl="F (goal_green_open)",
                                    #   sc_ltl="F (floor_purple_open) & F (floor_green_open) & (!(lava_red_open) U (floor_green_open)) & (!(lava_red_open) U (floor_purple_open))",
                                      use_alias=False,
                                      plot=self.plot_dfa)

    # over ride method to add the attribute self.get_iros_ts
    def _build_product(self):
        self._product_automaton = graph_factory.get('ProductGraph',
                                                    graph_name='product_automaton',
                                                    config_yaml='/config/product_automaton',
                                                    trans_sys=self._trans_sys,
                                                    automaton=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
                                                    absorbing=True,
                                                    finite=self.finite,
                                                    plot=self.plot_product,
                                                    integrate_accepting=True)

    @property
    def wombats_minigrid_TS(self):
        return self._wombats_minigrid_TS


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

        all_problems = False
        if all_problems:
            self._use_trans_sys_weights = True
            self._ts_config = "/config/Game_all_problems"
            self._auto_config = "/config/PDFA_onegoal"
        else:
            self._ts_config = "/config/Game_simple_loop"
            self._use_trans_sys_weights = False
            self._auto_config = "/config/PDFA_twogoals"

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
            graph_name='product_automaton',
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


class FrankaAbstractionGraph(GraphInstanceConstructionBase):

    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _build_graph_from_yaml(self):
        if self._trans_sys._graph_yaml is None:
            warnings.warn("Please ensure that you have first loaded the config data. You can do this by"
                          "setting the respective True in the builder instance.")

        _nodes = self._trans_sys._graph_yaml['nodes']
        _start_state = self._trans_sys._graph_yaml['start_state']

        # each node has an atomic proposition and a player associated with it. Some states also init and
        # accepting attributes associated with them
        for _n, _n_attr in _nodes.items():
            state_name = _n
            ap = _n_attr.get('observation')

            self._trans_sys.add_state(state_name, ap=ap, player="eve")

        self._trans_sys.add_initial_state(_start_state)

        _edges = self._trans_sys._graph_yaml['edges']

        for _u, _v in _edges.items():
            if _v is not None:
                for _n, _n_attr in _v.items():
                    _action = _n_attr.get('symbols')
                    self._trans_sys.add_edge(_u, _n, weight="-1", actions=_action)

    def _build_ts(self):
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=None,
                                            graph_name="trans_sys",
                                            config_yaml="/config/franka_abs",
                                            pre_built=False,
                                            from_file=True,
                                            built_in_ts_name="",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=0,
                                            finite=self.finite,
                                            plot_raw_ts=False)

        self._build_graph_from_yaml()
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=self._trans_sys,
                                            graph_name="trans_sys",
                                            config_yaml="/config/franka_abs",
                                            pre_built=False,
                                            from_file=False,
                                            built_in_ts_name="",
                                            save_flag=True,
                                            debug=False,
                                            plot=self.plot_ts,
                                            human_intervention=0,
                                            finite=self.finite,
                                            plot_raw_ts=False)
        # self._trans_sys.fancy_graph()

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="/config/automaton",
                                      save_flag=True,
                                      sc_ltl="F(p04p10p22)",
                                      use_alias=False,
                                      plot=self.plot_dfa)


def infinite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                                mini_grid_instance: Optional[MinigridGraph] = None,
                                epsilon: float = 0,
                                max_human_interventions: int = 5,
                                go_fast: bool = True,
                                finite: bool = False,
                                plot: bool = False,
                                plot_only_eve: bool = False):
    payoff = payoff_factory.get("cumulative", graph=trans_sys)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys, payoff)

    reg_syn_handle.infinite_reg_solver(minigrid_instance=mini_grid_instance,
                                       plot=plot,
                                       plot_only_eve=plot_only_eve,
                                       simulate_minigrid=False,
                                       go_fast=go_fast,
                                       finite=finite,
                                       epsilon=epsilon,
                                       max_human_interventions=max_human_interventions)


def compute_bounded_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                                epsilon: float = 0,
                                energy_bound: int = 0,
                                max_human_interventions: int = 5,
                                mini_grid_instance: Optional[MinigridGraph] = None,
                                debug: bool = False,
                                print_str: bool = False):

    iros_solver = IrosStrSolver(game=trans_sys, energy_bound=energy_bound, plot_modified_game=False)
    _start_state = trans_sys.get_initial_states()[0][0]
    if iros_solver.solve(debug=debug):
        print(f"There EXISTS a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")
        controls = iros_solver.get_controls_from_str_minigrid(epsilon=epsilon,
                                                              debug=debug,
                                                              max_human_interventions=max_human_interventions)
        mini_grid_instance.execute_str(controls)

    else:
        print(f"There DOES NOT exists a winning strategy from the  initial game state {_start_state} "
              f"with max cost of {iros_solver.str_map[_start_state]['cost']}")

    if print_str:
        iros_solver.print_map_dict()


def compute_winning_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                        mini_grid_instance: Optional[MinigridGraph] = None,
                        epsilon: float = 0,
                        max_human_interventions: int = 5,
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
        control = reachability_game_handle.get_pos_sequences(debug=False,
                                                             epsilon=epsilon,
                                                             max_human_interventions=max_human_interventions)
        mini_grid_instance.execute_str(_controls=control)
    else:
        print("Assuming Env to be adversarial, sys CANNOT force a visit to the accepting states")


def finite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid], mini_grid_instance: Optional[MinigridGraph] = None,
                              epsilon: float = 0,
                              max_human_interventions: int = 5,
                              plot: bool = False,
                              compute_reg_for_human: bool = False):
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

    payoff = payoff_factory.get("cumulative", graph=trans_sys)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys, payoff)

    # if mini_grid_instance:
    #     reg_syn_handle.add_common_accepting_state(plot=False)

    # reg_syn_handle.target_weighted_arena_finite_reg_solver(twa_graph=trans_sys,
    #                                                        debug=False,
    #                                                        purge_states=True,
    #                                                        plot_w_vals=True,
    #                                                        plot=plot,
    #                                                        plot_only_eve=False)

    reg_syn_handle.edge_weighted_arena_finite_reg_solver(minigrid_instance=mini_grid_instance,
                                                         purge_states=True,
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
    save_before_deleting_loops: bool = True,
    save_after_deleting_loops: bool = True):

    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()

    prism_interface = PrismInterfaceForTwoPlayerGame(use_docker=True)

    # Before Deleting Loops
    if save_before_deleting_loops:
        prism_interface.export_files_to_prism(trans_sys)

    # After Deleting Loops
    if save_after_deleting_loops:
        # TODO: Make an construction or copy function that outputs a new instanc w/o loops
        trans_sys_wo_loops = copy.deepcopy(trans_sys)
        trans_sys_wo_loops.delete_cycles(reachability_game_handle.sys_winning_region)
        trans_sys_wo_loops._graph_name = trans_sys._graph_name + '_wo_loops'
        trans_sys_wo_loops._graph.name = trans_sys._graph.name + '_wo_loops'
        trans_sys_wo_loops.plot_graph()

    prism_interface.run_prism(trans_sys_wo_loops, pareto=True, paretoepsilon=0.00001)
    print('Strategy')
    print(prism_interface.strategy)
    print(prism_interface.strategy_plan)
    print(prism_interface.strategy_trajectory)
    print(prism_interface.optimal_weights)
    print(prism_interface.pareto_points)

    solver = MultiObjectiveSolver(trans_sys)
    solver.solve(plot=plot)


def pure_adversarial_game(**kwargs):
    pure_game(cooperative=False, **kwargs)


def pure_cooperative_game(**kwargs):
    pure_game(cooperative=True, **kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=int, default=0,
        help="0 - the best strategy (for human too) and 1 - Completely random")
    parser.add_argument("--iros_flag", action="store_true", default=False,
        help="Whether to run experiements for IROS")
    parser.add_argument("--energy-bound", type=int, default=30,
        help="Energy Bound")
    parser.add_argument("--allowed_human_interventions", type=int, default=2,
        help="No. of times allowed for humans to intervene")
    parser.add_argument("--finite", action="store_true", default=False,
        help="")
    parser.add_argument("--go_fast", action="store_true", default=False,
        help="")
    parser.add_argument("--ts", type=str, default='two_goal',
        help="Choose which layer to add entropy (top, bottom, both, or None)",
        choices=['gym_minigrid', 'three_state', 'five_state', 'two_goal',
                 'variant_1_paper', 'target_weighted_arena', 'franka_abs'])
    parser.add_argument("--solver", type=str, default='pure_adv',
        help="Choose which layer to add entropy (top, bottom, both, or None)",
        choices=['finite_reg_synthesis', 'infinite_reg_synthesis', 'adversarial_game',
        'iros_str_synthesis', 'pure_adv', 'pure_coop'])
    parser.add_argument("--plot_ts", action="store_true", default=False,
        help="")
    parser.add_argument("--plot_dfa", action="store_true", default=False,
        help="")
    parser.add_argument("--plot_prod", action="store_true", default=False,
        help="")
    parser.add_argument("--plot", action="store_true", default=True,
        help="")
    parser.add_argument("--plot_auto_graph", action="store_true", default=False,
        help="")
    parser.add_argument("--plot_trans_graph", action="store_true", default=False,
        help="")
    parser.add_argument("--prune", action="store_true", default=False,
        help="")
    parser.add_argument("--multiple_accepting", action="store_true", default=False,
        help="")
    parser.add_argument("--integrate_accepting", type=str, default='only_accepts',
        help="",
        choices=['only_accepts', 'include_absorbs'])
    parser.add_argument("--weighting", type=str, default='automatonOnly',
        help="",
        choices=['linear', 'weightedlinear', 'automatonOnly'])
    # See how to pass a list as an argument
    # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser.add_argument("-p", "--complete_graph_players", nargs='+', default=['adam'],
        help="",
        choices=['eve', 'adam'])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    minigrid_instance = None

    integrate_only_accepts =  True if args.integrate_accepting == 'only_accepts' else False
    integrate_absorbs =  True if args.integrate_accepting == 'include_absorbs' else False

    ts_kwargs = {'_finite': args.finite,
                 '_plot_ts': args.plot_ts,
                 '_plot_dfa': args.plot_dfa,
                 '_plot_prod': args.plot_prod}

    # build the graph G on which we will compute the regret minimizing strategy
    if args.ts == 'gym_minigrid':
        ts = MinigridGraph(_iros_ts=args.iros_flag, _plot_minigrid=False, **ts_kwargs)
        minigrid_instance = ts
    elif args.ts == 'three_state':
        ts = ThreeStateExample(**ts_kwargs)
    elif args.ts == 'five_state':
        ts = FiveStateExample(**ts_kwargs)
    elif args.ts == 'two_goal':
        ts = TwoGoalsExample(plot_auto_graph=args.plot_auto_graph,
                             plot_trans_graph=args.plot_trans_graph,
                             prune=args.prune,
                             weighting=args.weighting,
                             complete_graph_players=args.complete_graph_players,
                             integrate_accepting=integrate_only_accepts,
                             multiple_accepting=args.multiple_accepting,
                             **ts_kwargs)
    elif args.ts == 'variant_1_paper':
        ts = VariantOneGraph(**ts_kwargs)
    elif args.ts == 'target_weighted_arena':
        ts = EdgeWeightedArena(_graph_type="ewa", **ts_kwargs)
    elif args.ts == 'franka_abs':
        ts = FrankaAbstractionGraph(**ts_kwargs)
    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    trans_sys = ts.product_automaton

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    solver_kwargs = {'trans_sys': trans_sys,
                     'mini_grid_instance': minigrid_instance,
                     'epsilon': args.epsilon,
                     'max_human_interventions': args.allowed_human_interventions}

    if args.solver == 'finite_reg_synthesis':
        finite_reg_minimizing_str(plot=args.plot,
                                  compute_reg_for_human=False,
                                  **solver_kwargs)
    elif args.solver == 'infinte_reg_synthesis':
        infinite_reg_minimizing_str(go_fast=args.go_fast,
                                    finite=args.finite,
                                    plot=args.plot,
                                    plot_only_eve=False,
                                    **solver_kwargs)
    elif args.solver == 'adversarial_game':
        compute_winning_str(debug=True,
                            print_winning_regions=False,
                            print_str=False,
                            **solver_kwargs)
    elif args.solver == 'iros_str_synthesis':
        compute_bounded_winning_str(energy_bound=args.energy_bound,
                                    debug=False,
                                    print_str=False,
                                    **solver_kwargs)
    elif args.solver == 'pure_adv':
        pure_adversarial_game(plot=args.plot,
                              compute_reg_for_human=False,
                              integrate_accepting=integrate_absorbs,
                              **solver_kwargs)
    elif args.solver == 'pure_coop':
        pure_cooperative_game(plot=args.plot,
                              compute_reg_for_human=False,
                              integrate_accepting=integrate_absorbs,
                              **solver_kwargs)
    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)
