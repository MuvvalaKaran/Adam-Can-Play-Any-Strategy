from src.graph import graph_factory

if __name__ == "__main__":
    # built a two player graph manually
    # two_player_graph = graph_factory.get("TwoPlayerGraph",
    #                                      graph_name="two_player_graph",
    #                                      config_yaml="config/two_player_graph",
    #                                      save_flag=True,
    #                                      pre_built=True,
    #                                      plot=True)
    # two_player_graph.plot_graph()

    # build the transition system
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="five_state_ts",
                                  save_flag=True,
                                  debug=True,
                                  plot=True,
                                  human_intervention=1,
                                  plot_raw_ts=False)
    #
    # # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=True,
                            sc_ltl="Fi & (!d U g)",
                            use_alias=False,
                            plot=True)
    #
    # # build the product automaton
    prod = graph_factory.get('ProductGraph',
                             graph_name='product_automaton',
                             config_yaml='config/product_automaton',
                             trans_sys=trans_sys,
                             dfa=dfa,
                             save_flag=True,
                             prune=False,
                             debug=False,
                             absorbing=False,
                             plot=True)
    # # build gmin
    gmin = graph_factory.get('GMin', graph=prod,
                             graph_name='gmin',
                             config_yaml='config/gmin',
                             debug=False,
                             save_flag=True,
                             plot=True)
    #
    gmax = graph_factory.get('GMax', graph=trans_sys,
                             graph_name='gmax',
                             config_yaml='config/gmax',
                             debug=False,
                             save_flag=True,
                             plot=True)

    # g.plot_graph()