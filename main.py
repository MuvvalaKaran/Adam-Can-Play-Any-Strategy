import gym
import enum
import abc
import warnings
import sys


from typing import Tuple, Optional, Dict

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
from src.strategy_synthesis import RegMinStrSyn
from helper_methods import run_save_output_mpg

# directory where we will be storing all the configuration files related to graphs
DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/config/"

"""
Note: When you create a NxN world, two units of width and height are consumed for drawing the boundary.
So a 4x4 world will be a 2x2 env and 5x5 will be a 3x3 env respectively.  
"""

class MiniGridEmptyEnv(enum.Enum):
    env_3 = 'MiniGrid-Empty-3x3-v0'
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


class GraphInstanceContructionBase(abc.ABC):

    def __init__(self, _finite: bool):
        self.finite = _finite
        self._trans_sys: Optional[FiniteTransSys] = None
        self._dfa: Optional[DFAGraph] = None
        self._product_automaton: Optional[ProductAutomaton] = None
        self._wombats_minigrid_TS: Optional[MinigridTransitionSystem] = None
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
                                                    dfa=self._dfa,
                                                    save_flag=True,
                                                    prune=False,
                                                    debug=False,
                                                    absorbing=True,
                                                    finite=self.finite,
                                                    plot=False)

    @property
    def product_automaton(self):
        return self._product_automaton

    @property
    def wombats_minigrid_TS(self):
        return self._wombats_minigrid_TS


class MinigridGraph(GraphInstanceContructionBase):

    def __init__(self, _finite: bool):
        super().__init__(_finite=_finite)

    def __get_TS_from_wombats(self) -> Tuple[MiniGrid, MinigridTransitionSystem]:
        # ENV_ID = 'MiniGrid-LavaComparison_noDryingOff-v0'
        # ENV_ID = 'MiniGrid-AlternateLavaComparison_AllCorridorsOpen-v0'
        ENV_ID = 'MiniGrid-DistShift1-v0'
        # ENV_ID = 'MiniGrid-LavaGapS5-v0'
        # ENV_ID = 'MiniGrid-Empty-5x5-v0'
        # ENV_ID = MiniGridEmptyEnv.env_5.value
        # ENV_ID = MiniGridLavaEnv.env_6.value

        env = gym.make(ENV_ID)
        env = StaticMinigridTSWrapper(env, actions_type='simple_static')
        env.render()

        wombats_minigrid_TS = active_automata.get(automaton_type='TS',
                                                  graph_data=env,
                                                  graph_data_format='minigrid')

        # file t0 dump the TS corresponding to the gym env
        file_name = ENV_ID + '_TS'
        wombats_minigrid_TS.to_yaml_file(DIR + file_name + ".yaml")

        regret_minigrid_TS = graph_factory.get('MiniGrid',
                                               graph_name="minigrid_TS",
                                               config_yaml=f"config/{file_name}",
                                               save_flag=True,
                                               plot=True)

        regret_minigrid_TS.build_graph_from_file()

        return regret_minigrid_TS, wombats_minigrid_TS

    def execute_str(self, _controls):
        self._wombats_minigrid_TS.run(_controls, record_video=True)

    def _build_ts(self):
        raw_trans_sys, self._wombats_minigrid_TS = self.__get_TS_from_wombats()

        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=raw_trans_sys,
                                            graph_name="trans_sys",
                                            config_yaml="config/trans_sys",
                                            pre_built=False,
                                            built_in_ts_name="",
                                            save_flag=True,
                                            debug=False,
                                            plot=False,
                                            human_intervention=0,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="config/automaton",
                                      save_flag=True,
                                      sc_ltl="!(lava_red_open) U (goal_green_open)",
                                      use_alias=False,
                                      plot=False)


class VariantOneGraph(GraphInstanceContructionBase):

    def __init__(self, _finite: bool):
        super().__init__(_finite=_finite)

    def _build_dfa(self):
        pass

    def _build_ts(self):
        pass

    def _build_product(self):
        self._product_automaton = graph_factory.get("TwoPlayerGraph",
                                                    graph_name="two_player_graph",
                                                    config_yaml="config/two_player_graph",
                                                    save_flag=True,
                                                    pre_built=True,
                                                    plot=False)


class ThreeStateExample(GraphInstanceContructionBase):

    def __init__(self, _finite: bool):
        super().__init__(_finite=_finite)

    def _build_ts(self):
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=None,
                                            graph_name="trans_sys",
                                            config_yaml="config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="three_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=False,
                                            human_intervention=1,
                                            finite=self.finite,
                                            plot_raw_ts=False)

    def _build_dfa(self):
        self._dfa = graph_factory.get('DFA',
                                      graph_name="automaton",
                                      config_yaml="config/automaton",
                                      save_flag=True,
                                      sc_ltl="!b U c",
                                      use_alias=False,
                                      plot=False)


class FiveStateExample(GraphInstanceContructionBase):

    def __init__(self, _finite: bool):
        super().__init__(_finite=_finite)

    def _build_ts(self):
        self._trans_sys = graph_factory.get('TS',
                                            raw_trans_sys=None,
                                            graph_name="trans_sys",
                                            config_yaml="config/trans_sys",
                                            pre_built=True,
                                            built_in_ts_name="five_state_ts",
                                            save_flag=True,
                                            debug=False,
                                            plot=False,
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
                                      plot=False)


if __name__ == "__main__":

    finite = False
    go_fast = True
    gym_minigrid = False
    three_state_ts = False
    five_state_ts = False
    variant_1_paper = True

    if gym_minigrid:
        miniGrid_instance = MinigridGraph(_finite=finite)
        trans_sys = miniGrid_instance.product_automaton
        wombats_minigrid_TS = miniGrid_instance.wombats_minigrid_TS

    elif three_state_ts:
        three_state_ts_instance = ThreeStateExample(_finite=finite)
        trans_sys = three_state_ts_instance.product_automaton

    elif five_state_ts:
        five_state_ts = FiveStateExample(_finite=finite)
        trans_sys = five_state_ts.product_automaton

    elif variant_1_paper:
        variant_1_instance = VariantOneGraph(_finite=finite)
        trans_sys = variant_1_instance.product_automaton

    else:
        warnings.warn("Please ensure at-least one of the flags is True")
        sys.exit(-1)

    # build the payoff function
    payoff = payoff_factory.get("mean", graph=trans_sys)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(trans_sys, payoff)

    if finite:
        w_prime = reg_syn_handle.compute_W_prime_finite(multi_thread=go_fast)
    else:
        w_prime = reg_syn_handle.compute_W_prime(multi_thread=go_fast)

    g_hat = reg_syn_handle.construct_g_hat(w_prime, finite=finite)
    reg_dict = run_save_output_mpg(g_hat, "g_hat", go_fast=True, debug=False)
    # g_hat.plot_graph()
    org_str = reg_syn_handle.plot_str_from_mgp(g_hat, reg_dict, only_eve=False, plot=True)

    if gym_minigrid:
        # map back str to minigrid env
        controls = reg_syn_handle.get_controls_from_str(org_str)
        miniGrid_instance.execute_str(_controls=controls)