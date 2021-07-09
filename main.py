import gym
import enum
import abc
import warnings
import sys
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mping

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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

from src.mpg_tool import MpgToolBox

# assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"

# directory where we will be storing all the configuration files related to graphs
DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/config/"

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

    def __init__(self, _finite: bool, _plot_ts: bool, _plot_dfa: bool, _plot_prod: bool):
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
                                                    dfa=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
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
        ENV_ID = 'MiniGrid-Empty-5x5-v0'
        # ENV_ID = MiniGridEmptyEnv.env_6.value
        # ENV_ID = MiniGridLavaEnv.env_6.value

        env = gym.make(ENV_ID)
        env = StaticMinigridTSWrapper(env, actions_type='simple_static')
        env.render()

        wombats_minigrid_TS = active_automata.get(automaton_type='TS',
                                                  graph_data=env,
                                                  graph_data_format='minigrid')

        # file to dump the TS corresponding to the gym env
        file_name = ENV_ID + '_TS'
        wombats_minigrid_TS.to_yaml_file(DIR + file_name + ".yaml")

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
                                      use_alias=False,
                                      plot=self.plot_dfa)

    # over ride method to add the attribute self.get_iros_ts
    def _build_product(self):
        self._product_automaton = graph_factory.get('ProductGraph',
                                                    graph_name='product_automaton',
                                                    config_yaml='/config/product_automaton',
                                                    trans_sys=self._trans_sys,
                                                    dfa=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
                                                    iros_ts=self.get_iros_ts,
                                                    absorbing=True,
                                                    finite=self.finite,
                                                    plot=self.plot_product)

    @property
    def wombats_minigrid_TS(self):
        return self._wombats_minigrid_TS


class MultiToolPlanner(GraphInstanceConstructionBase):
    """
    A class that takes the product of multiple transition systems associated with each tool. Given a shared task in LTL
    we then compute a high level strategy associated with system of tools.
    """
    def __init__(self,
                 _finite: bool = False,
                 _plot_ts: bool = False,
                 _plot_ts_prod: bool = False,
                 _plot_dfa: bool = False,
                 _plot_prod: bool = False):
        self._trans_sys_1: Optional[FiniteTransSys] = None
        self._trans_sys_2: Optional[FiniteTransSys] = None
        self.plot_ts_prod: bool = _plot_ts_prod
        super().__init__(_finite=_finite, _plot_ts=_plot_ts, _plot_dfa=_plot_dfa, _plot_prod=_plot_prod)

    def _build_ts(self):
        self._trans_sys_1 = graph_factory.get('TS',
                                              raw_trans_sys=None,
                                              graph_name="sun_imaging_trans_sys",
                                              config_yaml="/config/sun_imaging_trans_sys",
                                              pre_built=True,
                                              built_in_ts_name="sun_imaging_ts",
                                              save_flag=True,
                                              debug=False,
                                              plot=self.plot_ts,
                                              human_intervention=1,
                                              finite=self.finite,
                                              plot_raw_ts=False)

        self._trans_sys_2 = graph_factory.get('TS',
                                              raw_trans_sys=None,
                                              graph_name="camera_imaging_trans_sys",
                                              config_yaml="/config/camera_imaging_trans_sys",
                                              pre_built=True,
                                              built_in_ts_name="camera_imaging_ts",
                                              save_flag=True,
                                              debug=False,
                                              plot=self.plot_ts,
                                              human_intervention=1,
                                              finite=self.finite,
                                              plot_raw_ts=False)

        # this is a static method
        self._trans_sys = FiniteTransSys.compose_transition_systems(trans_sys_1=self._trans_sys_1,
                                                                    trans_sys_2=self._trans_sys_2,
                                                                    graph_name="product_ts",
                                                                    config_yaml="/config/product_ts",
                                                                    save_flag=True,
                                                                    debug=False,
                                                                    plot=self.plot_ts_prod)

    def _build_dfa(self):
        # _survey_fr = "G(!s) & F(a & Fb & Fc & F(d & Fe & Ff & F(g & Fh & Fi)))"
        _survey_fr = "G(!s) & F(a & Fb & F(c & Fd & F(e & Ff & F(g & Fh & Fi))))"

        _updated_fr = "F(con & X(son & F(!con & !son & de & X(!con & !son & don &" \
                      " F(!de & !don & con & X(!de & !don & son))))))"

        _updated_fr_w_safety = "F(con & X(son & F(!con & !son & de & X(!con & !son & don &" \
                      " F(!de & !don & con & X(!de & !don & son)))))) & G(con -> !(de & don))"

        _updated_fr_w_safety_1 = "F(con & X(son & F(de & X(don & F(con & X(son)))))) &" \
                                 " G((con | son)-> !(de & don)) &" \
                                 " G((de | don) -> !(con & son))"

        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="/config/automaton",
                                      save_flag=True,
                                      # sc_ltl=_updated_fr_w_safety,
                                      sc_ltl=_survey_fr,
                                      use_alias=False,
                                      plot=self.plot_dfa)

    def _build_product(self):
        self._product_automaton = graph_factory.get('ProductGraph',
                                                    graph_name='product_automaton',
                                                    config_yaml='/config/product_automaton',
                                                    trans_sys=self._trans_sys,
                                                    dfa=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
                                                    absorbing=True,
                                                    finite=self.finite,
                                                    plot=self.plot_product)


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
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="/config/automaton",
                                      save_flag=True,
                                      sc_ltl="F c",
                                      use_alias=False,
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
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="/config/automaton",
                                      save_flag=True,
                                      sc_ltl="!d U g",
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


class PlotterClass:

    def __init__(self, fig_title, robo_file_path=None, xlabel=None, ylabel=None):
        self.fig = None
        self.ax = None
        self.fig_title = fig_title
        self.xlabel = xlabel
        self.ylabel = ylabel

        # initialize a figure an axes handle
        self._create_ax_fig()
        if robo_file_path is not None:
            self.robot_patch = mping.imread(robo_file_path)

    def _create_ax_fig(self):
        self.fig = plt.figure(num=self.fig_title)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def plot_gridworld(self, print_grid_num=False, fontsize=None,  *args):
        # the env here the original env of size (N x M)
        # fig_title = args[0]

        cmax = 3
        rmax = 3

        # flag to print the grid numbers
        if print_grid_num:
            # we need to offset both the x and y to print in the center of the grid
            for x in range(cmax):
                for y in range(rmax):
                    off_x, off_y = x + 0.5, y + 0.5
                    self.plot_grid_num((off_x, off_y), value=f"{x, y}", fontsize=fontsize)

        # the location of the x_ticks is the middle of the grid
        def offset_ticks(x, offset=0.5):
            return x + offset

        # ticks = locations of the ticks and labels = label of the ticks
        self.ax.set_xticks(ticks=list(map(offset_ticks, range(cmax))), minor=False)
        self.ax.set_xticklabels(labels=range(cmax))
        self.ax.set_yticks(ticks=list(map(offset_ticks, range(rmax))), minor=False)
        self.ax.set_yticklabels(labels=range(rmax))

        # add points for gridline plotting
        self.ax.set_xticks(ticks=range(cmax), minor=True)
        self.ax.set_yticks(ticks=range(rmax), minor=True)

        # set x and y limits to adjust the view
        self.ax.set_xlim(left=0, right=cmax)
        self.ax.set_ylim(bottom=0, top=rmax)

        # set the gridlines at the minor xticks positions
        self.ax.yaxis.grid(True, which='minor')
        self.ax.xaxis.grid(True, which='minor')

    def plot_grid_num(self, xy, value, offset=None, **kwargs):
        """
        A method to add text to the grid world
        :param xy: (x, y) position of the text. No offset included in here
        :type xy: tuple
        :param value: The text that should go at each block
        :type value: basestring
        :param offset: how much to offset the values by in both x and y direction
        :type offset: int
        :param kwargs:
        :type kwargs:
        """
        x, y, = xy
        self.ax.annotate(value,
                         xy=(x, y),
                         xycoords='data',
                         horizontalalignment='center',
                         verticalalignment='center',
                         **kwargs)


def infinite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                                mini_grid_instance: Optional[MinigridGraph] = None,
                                epsilon: float = 0,
                                max_human_interventions: int = 5,
                                go_fast: bool = True,
                                finite: bool = False,
                                plot_result: bool = False,
                                plot_result_only_eve: bool = False):
    payoff = payoff_factory.get("cumulative", graph=trans_sys)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys, payoff)

    reg_syn_handle.infinite_reg_solver(minigrid_instance=mini_grid_instance,
                                       plot=plot_result,
                                       plot_only_eve=plot_result_only_eve,
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

def add_patch(ax, shape, xy, color='green', robo_image='sun_img.png', action=None, alpha=0.5):
    x, y = xy
    shape_case = {
        'circle': 1,
        'sun': 2
    }

    if shape_case[shape] == 1:
        circle = patches.Circle(
            (x, y),
            radius=0.2,
            color=color,
            alpha=1  # transparency value
        )
        ax.add_patch(circle)

        return circle

    elif shape_case[shape] == 2:
        robo_image = plt.imread(robo_image)
        imageBox = OffsetImage(robo_image, zoom=0.1)
        ab = AnnotationBbox(imageBox, xy, frameon=False)
        ax.add_artist(ab)

        return ab
    else:
        return warnings.warn("Not a valid shape. Need to add that patch to the _add_patch function")


def compute_multi_tool_planning(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                                debug: bool = False,
                                print_winning_regions: bool = False,
                                print_str: bool = False):
    """
    A function to compute a high level strategy for multiple tools given a common (shared/dependent) task
    :param trans_sys:
    :param debug:
    :param print_winning_regions:
    :param print_str:
    :return:
    """
    reachability_game_handle = ReachabilitySolver(game=trans_sys, debug=debug)
    reachability_game_handle.reachability_solver()
    _sys_str_dict = reachability_game_handle.sys_str
    _env_str_dict = reachability_game_handle.env_str

    final_plot = PlotterClass(fig_title="Apriori imaging", xlabel='x', ylabel='y')
    final_plot.plot_gridworld(print_grid_num=True, fontsize=10)

    def offset_ticks(x, offset=0.5):
        return x + offset

    robo_state_pos_map = {
        'r1': (0, 0),
        'r2': (1, 0),
        'r3': (2, 0),
        'r4': (0, 1),
        'r5': (1, 1),
        'r6': (2, 1),
        'r7': (0, 2),
        'r8': (1, 2),
        'r9': (2, 2),
    }

    sun_state_pos_map = {
        's1': (0, 0),
        's2': (1, 0),
        's3': (2, 0),
        's4': (0, 1),
        's5': (1, 1),
        's6': (2, 1),
        's7': (0, 2),
        's8': (1, 2),
        's9': (2, 2),
    }

    if print_winning_regions:
        reachability_game_handle.print_winning_region()

    if print_str:
        # reachability_game_handle.print_winning_strategies()
        time_step = 0
        winning_str: dict = reachability_game_handle.sys_str
        _init_state = trans_sys.get_initial_states()
        # sun state, camera state
        _curr_state = _init_state[0][0]
        curr_sys_xy = robo_state_pos_map.get(_curr_state[0][1])
        curr_env_xy = sun_state_pos_map.get(_curr_state[0][0])
        _next_state = winning_str.get(_curr_state)
        action = trans_sys.get_edge_attributes(_curr_state, _next_state, "actions")
        print(action)

        sys_patch = add_patch(ax=final_plot.ax, shape='circle', xy=tuple(map(offset_ticks, curr_sys_xy)))
        env_patch = add_patch(ax=final_plot.ax, shape='sun', xy=tuple(map(offset_ticks, curr_env_xy)))

        plt.savefig(f"frames/grid_{time_step}.png", dpi=200)
        while _next_state != _curr_state:
            time_step += 1
            sys_patch.remove()

            _curr_state = _next_state
            curr_sys_xy = robo_state_pos_map.get(_curr_state[0][1])
            curr_env_xy = sun_state_pos_map.get(_curr_state[0][0])
            sys_patch = add_patch(ax=final_plot.ax, shape='circle', xy=tuple(map(offset_ticks, curr_sys_xy)))
            plt.savefig(f"frames/grid_{time_step}_1.png", dpi=200)

            env_patch.remove()
            env_patch = add_patch(ax=final_plot.ax, shape='sun', xy=tuple(map(offset_ticks, curr_env_xy)))
            plt.savefig(f"frames/grid_{time_step}_2.png", dpi=200)

            _next_state = winning_str.get(_curr_state)
            action = trans_sys.get_edge_attributes(_curr_state, _next_state, "actions")
            print(action)

    # reachability_game_handle.plot_winning_strategy()


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


def finite_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                              mini_grid_instance: Optional[MinigridGraph] = None,
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
    gym_minigrid = False
    three_state_ts = False
    five_state_ts = False
    variant_1_paper = False
    target_weighted_arena = False
    franka_abs = False
    multitool_abs = True

    # solver to call
    finite_reg_synthesis = False
    infinte_reg_synthesis = False
    adversarial_game = False
    iros_str_synthesis = False
    miniGrid_instance = None
    multitool_str_synthesis = True

    # build the graph G on which we will compute the regret minimizing strategy
    if gym_minigrid:
        miniGrid_instance = MinigridGraph(_finite=finite,
                                          _iros_ts=IROS_FLAG,
                                          _plot_minigrid=False,
                                          _plot_ts=False,
                                          _plot_dfa=False,
                                          _plot_prod=False)
        trans_sys = miniGrid_instance.product_automaton
        wombats_minigrid_TS = miniGrid_instance.wombats_minigrid_TS

    elif three_state_ts:
        three_state_ts_instance = ThreeStateExample(_finite=finite,
                                                    _plot_ts=False,
                                                    _plot_dfa=False,
                                                    _plot_prod=False)
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
                                      _plot_prod=False)
        trans_sys = twa_graph.product_automaton

    elif franka_abs:
        franka_instance = FrankaAbstractionGraph(_finite=finite)
        trans_sys = franka_instance.product_automaton

    elif multitool_abs:
        multitool_graph = MultiToolPlanner(_finite=finite,
                                           _plot_ts=False,
                                           _plot_dfa=False,
                                           _plot_prod=False,
                                           _plot_ts_prod=False)
        trans_sys = multitool_graph.product_automaton
    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        finite_reg_minimizing_str(trans_sys,
                                  miniGrid_instance,
                                  epsilon=EPSILON,
                                  max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                                  plot=False,
                                  compute_reg_for_human=False)
    elif infinte_reg_synthesis:
        infinite_reg_minimizing_str(trans_sys,
                                    miniGrid_instance,
                                    max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                                    go_fast=go_fast,
                                    epsilon=EPSILON,
                                    finite=finite,
                                    plot_result=False,
                                    plot_result_only_eve=False)
    elif multitool_str_synthesis:
        compute_multi_tool_planning(trans_sys,
                                    debug=False,
                                    print_winning_regions=False,
                                    print_str=True)
    elif adversarial_game:
        compute_winning_str(trans_sys,
                            miniGrid_instance,
                            max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                            debug=True,
                            epsilon=EPSILON,
                            print_winning_regions=False,
                            print_str=False)
    elif iros_str_synthesis:
        compute_bounded_winning_str(trans_sys,
                                    mini_grid_instance=miniGrid_instance,
                                    energy_bound=ENERGY_BOUND,
                                    max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                                    debug=False,
                                    print_str=False,
                                    epsilon=EPSILON)
    else:
        warnings.warn("Please make sure that you select at-least one solver.")
        sys.exit(-1)
