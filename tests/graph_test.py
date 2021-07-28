from ..src.graph import graph_factory

DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/config/"


def test_build_from_file(config_file_name):

    two_player_graph_from_file = graph_factory.get("TwoPlayerGraph",
                                                   graph_name="two_player_graph_from_file",
                                                   config_yaml=config_file_name,
                                                   save_flag=True,
                                                   from_file=True)

    two_player_graph_from_file.build_graph_from_file()
    two_player_graph_from_file.fancy_graph()


if __name__ == "__main__":
    # run_wombats()

    # built a two player graph manually
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="config/two_player_graph",
                                         save_flag=False,
                                         pre_built=False,
                                         plot=False)

    # build the transition system
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="/config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="five_state_ts",
                                  save_flag=False,
                                  debug=True,
                                  plot=False,
                                  human_intervention=1,
                                  plot_raw_ts=False)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="/config/automaton",
                            save_flag=False,
                            sc_ltl="Fi & (!d U g)",
                            use_alias=False,
                            plot=False)

    # build the product automaton
    prod = graph_factory.get('ProductGraph',
                             graph_name='product_automaton',
                             config_yaml='/config/product_automaton',
                             trans_sys=trans_sys,
                             dfa=dfa,
                             save_flag=False,
                             prune=False,
                             debug=False,
                             absorbing=False,
                             plot=False)
