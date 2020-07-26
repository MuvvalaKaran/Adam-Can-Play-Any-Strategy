import yaml
import gym

from src.graph import graph_factory
from wombats.systems import StaticMinigridTSWrapper
from wombats.automaton import MinigridTransitionSystem
from wombats.automaton import active_automata

DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/config/"


def _from_yaml(config_file_name):
    # allowed_exts = ['.yaml', '.yml']
    # if file_name in allowed_exts:
    with open(config_file_name, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.Loader)

    return config_data


def run_wombats():
    # ENV_ID = 'MiniGrid-LavaComparison_noDryingOff-v0'
    # ENV_ID = 'MiniGrid-AlternateLavaComparison_AllCorridorsOpen-v0'
    ENV_ID = 'MiniGrid-DistShift1-v0'
    # ENV_ID = 'MiniGrid-LavaGapS5-v0'
    env = gym.make(ENV_ID)

    env = StaticMinigridTSWrapper(env, actions_type='static')
    # env.render_notebook()
    minigrid_TS = active_automata.get(automaton_type='TS', graph_data=env,
                                      graph_data_format='minigrid')
    # minigrid_TS.draw('_'.join(["", ENV_ID, 'TS']))
    file_name = ENV_ID + "_TS.yaml"
    minigrid_TS.to_yaml_file(DIR+file_name)

    # ts = graph_factory.get("TS",
    #                        graph_name=file_name,
    #                        config_yaml=DIR+file_name,color=("lightgrey", "red", "purple")
    #                        raw_trans_sys=None,
    #                        from_file=True,
    #                        pre_built=False,
    #                        save_flag=True)

    graph_factory.get('MiniGrid',
                      graph_name=file_name,
                      config_yaml=DIR+file_name,
                      save_flag=True,
                      plot=True)

if __name__ == "__main__":
    run_wombats()

    # built a two player graph manually
    # two_player_graph = graph_factory.get("TwoPlayerGraph",
    #                                      graph_name="two_player_graph",
    #                                      config_yaml="config/two_player_graph",
    #                                      save_flag=True,
    #                                      pre_built=True,
    #                                      plot=True)
    # # two_player_graph.plot_graph()
    # DIR = "/home/karan-m/Documents/wombats/config/"
    # file_name = "PDFA_MiniGrid_synthesis_manual_spec.yaml"
    # _from_yaml(DIR+file_name)

    # build the transition system
    # trans_sys = graph_factory.get('TS',
    #                               raw_trans_sys=None,
    #                               graph_name="trans_sys",
    #                               config_yaml="config/trans_sys",
    #                               pre_built=True,
    #                               built_in_ts_name="five_state_ts",
    #                               save_flag=True,
    #                               debug=True,
    #                               plot=True,
    #                               human_intervention=1,
    #                               plot_raw_ts=False)
    # #
    # # # build the dfa
    # dfa = graph_factory.get('DFA',
    #                         graph_name="automaton",
    #                         config_yaml="config/automaton",
    #                         save_flag=True,
    #                         sc_ltl="Fi & (!d U g)",
    #                         use_alias=False,
    #                         plot=True)
    # #
    # # # build the product automaton
    # prod = graph_factory.get('ProductGraph',
    #                          graph_name='product_automaton',
    #                          config_yaml='config/product_automaton',
    #                          trans_sys=trans_sys,
    #                          dfa=dfa,
    #                          save_flag=True,
    #                          prune=False,
    #                          debug=False,
    #                          absorbing=False,
    #                          plot=True)
    # # # build gmin
    # gmin = graph_factory.get('GMin', graph=prod,
    #                          graph_name='gmin',
    #                          config_yaml='config/gmin',
    #                          debug=False,
    #                          save_flag=True,
    #                          plot=True)
    # #
    # gmax = graph_factory.get('GMax', graph=trans_sys,
    #                          graph_name='gmax',
    #                          config_yaml='config/gmax',
    #                          debug=False,
    #                          save_flag=True,
    #                          plot=True)

    # g.plot_graph()