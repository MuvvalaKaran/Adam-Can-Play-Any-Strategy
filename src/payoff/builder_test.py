from src.payoff import payoff_factory
from src.graph import graph_factory

if __name__ == "__main__":
    # build the transition system
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="three_state_ts",
                                  save_flag=False,
                                  debug=False,
                                  plot=True,
                                  human_intervention=1,
                                  plot_raw_ts=False)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=False,
                            sc_ltl="!b U c",
                            use_alias=False,
                            plot=True)

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
                             plot=True)

    limsup = payoff_factory.get("mean",
                                graph=prod)

    limsup.cycle_main()
    for k, v in limsup.loop_vals.items():
        print(f"Play: {k} : val: {v} ")
