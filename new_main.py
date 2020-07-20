# main script file that

from src.graph import graph_factory
from src.payoff import payoff_factory
from src.strategy_synthesis import RegMinStrSyn


if __name__ == "__main__":

    # build a graph
    trans_sys = graph_factory.get('TS',
                                  raw_trans_sys=None,
                                  graph_name="trans_sys",
                                  config_yaml="config/trans_sys",
                                  pre_built=True,
                                  built_in_ts_name="three_state_ts",
                                  save_flag=False,
                                  debug=False,
                                  plot=False,
                                  human_intervention=1)

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
                             save_flag=False,
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

    reg_dict = reg_syn_handle.compute_aval(g_hat, "", w_prime, optimistic=True)