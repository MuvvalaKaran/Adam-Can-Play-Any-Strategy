import gym
import enum
import abc
import warnings
import sys
import copy

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

assert ('linux' in sys.platform), "This code has been successfully tested in Linux-18.04 & 16.04 LTS"

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
        # ENV_ID = 'MiniGrid-Empty-5x5-v0'
        # ENV_ID = MiniGridEmptyEnv.env_5.value
        ENV_ID = MiniGridLavaEnv.env_3.value

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
                                            human_intervention=self.human_intervention,
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
                                      sc_ltl="!(lava_red_open) U (goal_green_open)",
                                      # sc_ltl="F (goal_green_open)",
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


def add_common_accepting_state(trans_sys: TwoPlayerGraph, plot: bool = False) -> TwoPlayerGraph:
    """
    A method that adds a auxiliary accepting state from the current accepting state as well as the trap state to
    the new accepting state. Visually

    trap--W_bar --> new_accp (self-loop weight 0)
             /^
            /
           0
          /
         /
        /
    Acc

    W_bar : Highest payoff any state could ever achieve when playing total-payoff game in cooperative setting,
    assuming strategies to be non-cylic, will be equal to (|V| -1)W where W is the max absolute weight.

    Now we remove Acc as the accepting state and add new_accp as the new accepting state and initialize this state
    to 0 in the value iteration algorithm - essentially eliminating a trap region.

    :return:
    """
    # remove the current accepting state
    _trans_sys = copy.deepcopy(trans_sys)

    old_accp_states = _trans_sys.get_accepting_states()
    trap_states = _trans_sys.get_trap_states()
    _num_of_nodes = len(list(_trans_sys._graph.nodes))
    _W = abs(_trans_sys.get_max_weight())
    w_bar = ((_num_of_nodes - 1) * _W)

    # remove self-loops of accepting states and add edge from that state to the new accepting state with edge
    # weight 0
    for _accp in old_accp_states:
        _trans_sys._graph.remove_edge(_accp, _accp)
        _trans_sys.remove_state_attr(_accp, "accepting")
        _trans_sys.add_state_attribute(_accp, "player", "eve")
        _trans_sys.add_weighted_edges_from([(_accp, 'tmp_accp', 0)])

    # remove self-loops of trap states and add edge from that state to the new accepting state with edge weight
    # w_bar
    for _trap in trap_states:
        _trans_sys._graph.remove_edge(_trap, _trap)
        _trans_sys.add_state_attribute(_trap, "player", "eve")
        _trans_sys.add_weighted_edges_from([(_trap, 'tmp_accp', w_bar)])

    _trans_sys.add_weighted_edges_from([('tmp_accp', 'tmp_accp', 0)])
    _trans_sys.add_accepting_state('tmp_accp')
    _trans_sys.add_state_attribute('tmp_accp', "player", "eve")

    if plot:
        _trans_sys.plot_graph()

    return _trans_sys


def compute_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
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

    reg_syn_handle.infinte_reg_solver(minigrid_instance=mini_grid_instance,
                                      plot=plot_result,
                                      plot_only_eve=plot_result_only_eve,
                                      simulate_minigrid=False,
                                      go_fast=go_fast,
                                      finite=finite,
                                      epsilon=epsilon,
                                      max_human_interventions=max_human_interventions)

    # w_prime = reg_syn_handle.compute_W_prime(go_fast=go_fast, debug=False)
    #
    # # g_hat = reg_syn_handle.construct_g_hat(w_prime, game=trans_sys, finite=finite, debug=True,
    # #                                        plot=True)
    # g_hat = reg_syn_handle.construct_g_hat(w_prime, game=None, finite=finite, debug=True,
    #                                        plot=False)
    #
    # mpg_g_hat_handle = MpgToolBox(g_hat, "g_hat")
    #
    # reg_dict, reg_val = mpg_g_hat_handle.compute_reg_val(go_fast=True, debug=False)
    # # g_hat.plot_graph()
    # org_str = reg_syn_handle.plot_str_from_mgp(g_hat, reg_dict, only_eve=plot_result_only_eve, plot=plot_result)
    #
    # # map back str to minigrid env
    # if mini_grid_instance is not None:
    #     controls = reg_syn_handle.get_controls_from_str_minigrid(org_str,
    #                                                              epsilon=epsilon,
    #                                                              max_human_interventions=max_human_interventions)
    #     mini_grid_instance.execute_str(_controls=(reg_val, controls))
    # else:
    # control = reg_syn_handle.get_controls_from_str(org_str, debug=True)


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


def new_compute_reg_minimizing_str(trans_sys: Union[FiniteTransSys, TwoPlayerGraph, MiniGrid],
                                   mini_grid_instance: Optional[MinigridGraph] = None,
                                   epsilon: float = 0,
                                   max_human_interventions: int = 5,
                                   plot: bool = True,
                                   reg_syn: bool = False,
                                   alt_reg_syn: bool = False):
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

    reg_syn_handle.finite_reg_solver(minigrid_instance=mini_grid_instance,
                                     plot=plot,
                                     plot_only_eve=False,
                                     simulate_minigrid=True,
                                     epsilon=epsilon,
                                     max_human_interventions=max_human_interventions)

    # # Add auxiliary accepting state
    # _game, _ = add_common_accepting_state(trans_sys, plot=False)
    #
    # # play reg minimizing game
    # if reg_syn:
    #     # let try computing coop strs instead of alternate ones
    #     coop_mcr_solver = ValueIteration(_game, competitive=False)
    #     coop_mcr_solver.solve(debug=True, plot=False)
    #     coop_val_dict = coop_mcr_solver.state_value_dict
    #
    #     # compute competitive values from each state
    #     comp_mcr_solver = ValueIteration(_game, competitive=True)
    #     comp_mcr_solver.solve(debug=True, plot=False)
    #     _comp_val_dict = comp_mcr_solver.state_value_dict
    #
    #     comp_vals: Dict[Tuple: float] = {}
    #
    #     for edge in _game._graph.edges():
    #         _u = edge[0]
    #         _v = edge[1]
    #
    #         # if the node belongs to adam, then the corresponding edge is assigned -inf
    #         if _game._graph.nodes(data='player')[_u] == 'adam':
    #             comp_vals.update({edge: 0})
    #
    #         else:
    #             _state_cost = _comp_val_dict[_v]
    #             _curr_w = _game.get_edge_weight(_u, _v)
    #             _total_weight = _curr_w + _state_cost
    #
    #             comp_vals.update({edge: _total_weight})
    #
    #     # now that we comp and coop value for each edge(str), we construct a new graph with the edge weight given by
    #     # comp(sigma, tau) - coop(sigma', tau)
    #
    #     _adv_reg_game: TwoPlayerGraph = copy.deepcopy(_game)
    #
    #     for _e in _game._graph.edges():
    #         _u = _e[0]
    #         _v = _e[1]
    #
    #         if _game._graph.nodes(data='player')[_u] == 'adam':
    #             _new_weight = 0
    #             _adv_reg_game._graph[_u][_v][0]['weight'] = _new_weight
    #             continue
    #
    #         # if the out-degree of a sys state is exactly one then _new_weight is 0
    #         if _game._graph.out_degree(_u) == 1:
    #             _new_weight = 0
    #         else:
    #             # _new_weight = comp_vals[_e] - alt_coop_vals[_e]
    #             _new_weight = comp_vals[_e] - coop_val_dict[_u]
    #
    #         _adv_reg_game._graph[_u][_v][0]['weight'] = _new_weight
    #
    #     # now lets play a adversarial game on this new graph and compute reg minimizing strs
    #     adv_mcr_solver = ValueIteration(_adv_reg_game, competitive=True)
    #     str_dict = adv_mcr_solver.solve(debug=True, plot=False)
    #     trans_sys = _adv_reg_game
    #     # remove edges to the tmp_accp state for ease of plotting
    #     trans_sys._graph.remove_edge('trap', 'tmp_accp')
    #     trans_sys._graph.remove_edge('accept_all', 'tmp_accp')
    #     trans_sys.add_weighted_edges_from([('accept_all', 'accept_all', 0), ('trap', 'trap', 0)])
    #
    # elif alt_reg_syn:
    #     # compute the competitive strategy for the human player
    #     pass
    #
    # # play a pure adversarial game
    # else:
    #     mcr_solver = ValueIteration(_game, competitive=True)
    #     str_dict = mcr_solver.solve(debug=True, plot=False)
    #
    # if plot:
    #     # add str attr
    #     trans_sys.set_edge_attribute('strategy', False)
    #
    #     # add edges that belong to str as True (red for visualization)
    #     for curr_node, next_node in str_dict.items():
    #         if next_node == "tmp_accp":
    #             next_node = curr_node
    #
    #         if isinstance(next_node, list):
    #             for n_node in next_node:
    #                 trans_sys._graph.edges[curr_node, n_node, 0]['strategy'] = True
    #         else:
    #             trans_sys._graph.edges[curr_node, next_node, 0]['strategy'] = True
    #
    #     trans_sys.plot_graph()
    #
    # if mini_grid_instance:
    #     str_dict["accept_all"] = "accept_all"
    #     str_dict["T0_S2"] = "T0_S2"
    #     payoff = payoff_factory.get("cumulative", graph=trans_sys)
    #     reg_syn_handle = RegMinStrSyn(trans_sys, payoff)
    #
    #     controls = reg_syn_handle.get_controls_from_str_minigrid(str_dict,
    #                                                              epsilon=epsilon,
    #                                                              max_human_interventions=max_human_interventions)
    #     mini_grid_instance.execute_str(_controls=(0, controls))
    # sys.exit(-1)


if __name__ == "__main__":

    # define some constants
    EPSILON = 0  # 0 - the best strategy (for human too) and 1 - Completely random
    IROS_FLAG = False
    ENERGY_BOUND = 30
    ALLOWED_HUMAN_INTERVENTIONS = 0

    # some constants related to computation
    finite = True
    go_fast = True

    # some constants that allow for appr _instance creations
    gym_minigrid = True
    three_state_ts = False
    five_state_ts = False
    variant_1_paper = False
    franka_abs = False

    # solver to call
    finite_reg_synthesis = False
    infinte_reg_synthesis = True
    adversarial_game = False
    iros_str_synthesis = False
    miniGrid_instance = None

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
        five_state_ts = FiveStateExample(_finite=finite)
        trans_sys = five_state_ts.product_automaton

    elif variant_1_paper:
        variant_1_instance = VariantOneGraph(_finite=finite,
                                             _plot_prod=False)
        trans_sys = variant_1_instance.product_automaton

    elif franka_abs:
        franka_instance = FrankaAbstractionGraph(_finite=finite)
        trans_sys = franka_instance.product_automaton
    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    print(f"No. of nodes in the product graph is :{len(trans_sys._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(trans_sys._graph.edges())}")

    if finite_reg_synthesis:
        new_compute_reg_minimizing_str(trans_sys,
                                       miniGrid_instance,
                                       epsilon=EPSILON,
                                       max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                                       plot=False,
                                       reg_syn=True,
                                       alt_reg_syn=False)
    elif infinte_reg_synthesis:
        compute_reg_minimizing_str(trans_sys,
                                   miniGrid_instance,
                                   max_human_interventions=ALLOWED_HUMAN_INTERVENTIONS,
                                   go_fast=go_fast,
                                   epsilon=EPSILON,
                                   finite=finite,
                                   plot_result=False,
                                   plot_result_only_eve=False)
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