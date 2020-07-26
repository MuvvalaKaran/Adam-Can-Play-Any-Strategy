import gym
import enum

# import wombats packages
from wombats.systems import StaticMinigridTSWrapper
from wombats.automaton import active_automata

# import local packages
from src.graph import graph_factory
from src.graph import MiniGrid
from src.payoff import payoff_factory
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


def get_TS_from_wombats() -> MiniGrid:
    # ENV_ID = 'MiniGrid-LavaComparison_noDryingOff-v0'
    # ENV_ID = 'MiniGrid-AlternateLavaComparison_AllCorridorsOpen-v0'
    # ENV_ID = 'MiniGrid-DistShift1-v0'
    # ENV_ID = 'MiniGrid-LavaGapS5-v0'
    # ENV_ID = 'MiniGrid-Empty-5x5-v0'
    ENV_ID = MiniGridEmptyEnv.env_4.value

    env = gym.make(ENV_ID)
    env = StaticMinigridTSWrapper(env, actions_type='static')
    env.render()

    wombats_minigrid_TS = active_automata.get(automaton_type='TS',
                                              graph_data=env,
                                              graph_data_format='minigrid')

    # file t0 dump the TS corresponding to the gym env
    file_name = ENV_ID + '_TS'
    wombats_minigrid_TS.to_yaml_file(DIR + file_name + ".yaml")

    regret_minigrid_TS = graph_factory.get('MiniGrid',
                                           graph_name="minigris_TS",
                                           config_yaml=f"config/{file_name}",
                                           save_flag=True,
                                           plot=False)

    regret_minigrid_TS.build_graph_from_file()

    return regret_minigrid_TS


if __name__ == "__main__":

    finite = False
    # build the TS
    raw_trans_sys = get_TS_from_wombats()

    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=raw_trans_sys,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=False,
                                  built_in_ts_name="",
                                  save_flag=True,
                                  debug=False,
                                  plot=False,
                                  human_intervention=0,
                                  plot_raw_ts=False)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=True,
                            sc_ltl="F(goal_green_open)",
                            use_alias=False,
                            plot=False)

    # build the product automaton
    prod = graph_factory.get('ProductGraph',
                             graph_name='product_automaton',
                             config_yaml='config/product_automaton',
                             trans_sys=trans_sys,
                             dfa=dfa,
                             save_flag=True,
                             prune=False,
                             debug=False,
                             absorbing=True,
                             plot=False)

    # gmin = graph_factory.get('GMin', graph=prod,
    #                          graph_name="gmin",
    #                          config_yaml="config/gmin",
    #                          debug=False,
    #                          save_flag=False,
    #                          plot=True)
    #
    # game = graph_factory.get("TwoPlayerGraph",
    #                          graph_name="two_player_graph",
    #                          config_yaml="config/two_player_graph",
    #                          save_flag=True,
    #                          pre_built=True,
    #                          plot=True)

    # build the payoff function
    payoff = payoff_factory.get("mean", graph=prod)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(prod, payoff)

    if finite:
        w_prime = reg_syn_handle.compute_W_prime_finite()
    else:
        w_prime = reg_syn_handle.compute_W_prime()

    g_hat = reg_syn_handle.construct_g_hat(w_prime, acc_min_edge_weight=False)
    reg_dict = run_save_output_mpg(g_hat, "g_hat", go_fast=True)
    # g_hat.plot_graph()
    reg_syn_handle.plot_str_from_mgp(g_hat, reg_dict, only_eve=False, plot=True)