from src.graph import graph_factory
from src.payoff import payoff_factory
from src.strategy_synthesis import RegMinStrSyn
from helper_methods import create_mpg_file, read_mpg_op, run_save_output_mpg


if __name__ == "__main__":

    finite = False
    # build the TS
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="five_state_ts",
                                  save_flag=True,
                                  debug=False,
                                  plot=False,
                                  human_intervention=1,
                                  plot_raw_ts=False)

    # build the dfa
    dfa = graph_factory.get('DFA',
                            graph_name="automaton",
                            config_yaml="config/automaton",
                            save_flag=False,
                            sc_ltl="Fi & (!d U g)",
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
    reg_syn_handle.plot_str_from_mgp(g_hat, reg_dict, only_eve=True, plot=True)