# main script file that
from src.graph import graph_factory
from src.payoff import payoff_factory
from src.strategy_synthesis import RegMinStrSyn


if __name__ == "__main__":

    plot_all = False
    finite = False
    # build the TS
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="three_state_ts",
                                  save_flag=True,
                                  debug=False,
                                  plot=True,
                                  human_intervention=1,
                                  plot_raw_ts=True)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=False,
                            sc_ltl="!b U c",
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
                             plot=True)
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
    g_hat.plot_graph()

    reg_dict = reg_syn_handle.compute_aval(g_hat, w_prime,
                                           optimistic=False,
                                           plot_all=plot_all,
                                           bypass_implementation=True,
                                           print_reg=True,
                                           not_validity_check=True)

    if reg_dict != {}:
        if plot_all:
            reg_syn_handle.plot_all_str_g_hat(g_hat, reg_dict, only_eve=False, plot=True)
        else:
            reg_syn_handle.plot_str_g_hat(g_hat, reg_dict, only_eve=False, plot=True)
    # else:
    #     print(f"There does not exists reg below the threshold value you entered.")