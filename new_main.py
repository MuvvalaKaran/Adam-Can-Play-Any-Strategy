# main script file that
from src.graph import graph_factory
from src.payoff import payoff_factory
from src.strategy_synthesis import RegMinStrSyn


if __name__ == "__main__":

    # build the TS
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="five_state_ts",
                                  save_flag=False,
                                  debug=False,
                                  plot=False,
                                  human_intervention=1)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=False,
                            sc_ltl="!d U g",
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

    # build the payoff function
    payoff = payoff_factory.get("mean", graph=prod)

    # build an instance of strategy minimization class
    reg_syn_handle = RegMinStrSyn(prod, payoff)

    w_prime = reg_syn_handle.compute_W_prime()

    g_hat = reg_syn_handle.construct_g_hat(w_prime)
    # g_hat.plot_graph()

    plot_all = True
    reg_dict = reg_syn_handle.compute_aval(g_hat, w_prime, optimistic=False, plot_all=plot_all)
    if plot_all:
        reg_syn_handle.plot_all_str_g_hat(g_hat, reg_dict, only_eve=False, plot=True)
    else:
        reg_syn_handle.plot_str_g_hat(g_hat, reg_dict, only_eve=False, plot=True)