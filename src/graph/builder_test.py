from src.graph import graph

if __name__ == "__main__":
    # built a two player graph manually
    two_player_graph = graph.get("TwoPlayerGraph",
                                 graph_name="org_grpah",
                                 config_yaml="config/org_graph",
                                 save_flag=True,
                                 pre_built=True)
    two_player_graph.plot_graph()

    # build the transition system
    trans_sys = graph.get('TS',
                          raw_trans_sys=None,
                          graph_name="trans_sys",
                          config_yaml="config/trans_sys",
                          pre_built=True,
                          built_in_ts_name="three_state_ts",
                          save_flag=False,
                          debug=True,
                          plot=True,
                          human_intervention=1,
                          plot_raw_ts=False)

    # build the dfa
    dfa = graph.get('DFA',
                    graph_name="automaton",
                    config_yaml="config/automaton",
                    save_flag=True,
                    sc_ltl="!b U c",
                    use_alias=True,
                    plot=True)

    # build the product automaton


    # g = graph.get('GMin', graph=t,
    #               graph_name='gmin',
    #               config_yaml='config/gmin',
    #               trans_sys=f,
    #               manual_constr=False,
    #               debug=False,
    #               save_flag=True,
    #               plot=True,
    #               human_intervention=1,
    #               pre_built=False,
    #               plot_raw_ts=True)
    # g.plot_graph()