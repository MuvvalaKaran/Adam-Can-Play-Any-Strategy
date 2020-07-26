# a script to test infinte and finite payoff scripts are working fine or not
from src.payoff import payoff_factory
from src.graph import graph_factory
from src.strategy_synthesis import RegMinStrSyn

if __name__ == "__main__":

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="two_player_graph",
                                         config_yaml="config/two_player_graph",
                                         save_flag=True,
                                         pre_built=False,
                                         from_file=False,
                                         plot=False)

    two_player_graph.add_states_from(["v1", "v2", "v3", "v4", "v5"])
    two_player_graph.add_edge("v1", "v3", weight='1')
    two_player_graph.add_edge("v1", "v2", weight='1')
    two_player_graph.add_edge("v3", "v3", weight='0.5')
    two_player_graph.add_edge("v2", "v3", weight='3')
    two_player_graph.add_edge("v3", "v5", weight='1')
    two_player_graph.add_edge("v5", "v5", weight='1')
    two_player_graph.add_edge("v5", "v4", weight='1')
    two_player_graph.add_edge("v2", "v4", weight='2')
    two_player_graph.add_edge("v2", "v1", weight='-1')
    two_player_graph.add_edge("v4", "v1", weight='2')

    two_player_graph.add_initial_state("v1")
    two_player_graph.plot_graph()

    payoff_handle = payoff_factory.get("mean", graph=two_player_graph)
    reg_syn_handle = RegMinStrSyn(two_player_graph, payoff_handle)
    reg_syn_handle.compute_W_prime()
