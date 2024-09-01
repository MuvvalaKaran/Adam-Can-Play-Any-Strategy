from src.graph import graph_factory, TwoPlayerGraph


def admissibility_game_toy_example_1(plot: bool = False) -> TwoPlayerGraph:
    """
     The example from Figure of the AAAI 25 paper - 08/01/24
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="admissibile_game_1",
                                         config_yaml="/config/admissibile_game_1",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam)
    two_player_graph.add_states_from(["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"])

    two_player_graph.add_initial_state('v0')
    two_player_graph.add_state_attribute("v0", "player", "eve")
    two_player_graph.add_state_attribute("v1", "player", "adam")
    two_player_graph.add_state_attribute("v2", "player", "adam")
    two_player_graph.add_state_attribute("v3", "player", "eve")
    two_player_graph.add_state_attribute("v4", "player", "eve")
    two_player_graph.add_state_attribute("v5", "player", "adam")
    two_player_graph.add_state_attribute("v6", "player", "eve")
    two_player_graph.add_state_attribute("v7", "player", "adam")
    two_player_graph.add_state_attribute("v8", "player", "eve")
    two_player_graph.add_state_attribute("v9", "player", "eve")
    two_player_graph.add_state_attribute("v10", "player", "adam")

    two_player_graph.add_edge("v0", "v1", weight=1)
    two_player_graph.add_edge("v0", "v2", weight=1)
    two_player_graph.add_edge("v1", "v4", weight=0)
    two_player_graph.add_edge("v2", "v3", weight=0)
    two_player_graph.add_edge("v3", "v2", weight=1)
    two_player_graph.add_edge("v2", "v6", weight=0)
    two_player_graph.add_edge("v4", "v5", weight=9)
    two_player_graph.add_edge("v4", "v7", weight=1)
    two_player_graph.add_edge("v5", "v6", weight=0)
    two_player_graph.add_edge("v7", "v8", weight=0)
    two_player_graph.add_edge("v7", "v9", weight=0)
    two_player_graph.add_edge("v9", "v10", weight=1)
    two_player_graph.add_edge("v8", "v10", weight=8)
    two_player_graph.add_edge("v10", "v6", weight=0)

    two_player_graph.add_accepting_states_from(["v6"])

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


def admissibility_game_toy_example_2(plot: bool = False) -> TwoPlayerGraph:
    """
     The example from Figure 5 of the AAAI 25 paper - 08/01/24
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="admissibile_game_2",
                                         config_yaml="/config/admissibile_game_2",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam)
    two_player_graph.add_states_from(["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", \
                                       "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"])

    two_player_graph.add_initial_state('v0')
    two_player_graph.add_state_attribute("v0", "player", "eve")
    two_player_graph.add_state_attribute("v1", "player", "adam")
    two_player_graph.add_state_attribute("v2", "player", "adam")
    two_player_graph.add_state_attribute("v3", "player", "eve")
    two_player_graph.add_state_attribute("v4", "player", "eve")
    two_player_graph.add_state_attribute("v5", "player", "adam")
    two_player_graph.add_state_attribute("v6", "player", "adam")
    two_player_graph.add_state_attribute("v7", "player", "eve")
    two_player_graph.add_state_attribute("v8", "player", "adam")
    two_player_graph.add_state_attribute("v9", "player", "adam")
    two_player_graph.add_state_attribute("v10", "player", "adam")
    two_player_graph.add_state_attribute("v11", "player", "eve")
    two_player_graph.add_state_attribute("v12", "player", "eve")
    two_player_graph.add_state_attribute("v13", "player", "eve")
    # value nodes - acceptine states. 
    two_player_graph.add_state_attribute("v14", "player", "eve")
    two_player_graph.add_state_attribute("v15", "player", "eve")
    two_player_graph.add_state_attribute("v16", "player", "eve")
    two_player_graph.add_state_attribute("v17", "player", "eve")
    two_player_graph.add_state_attribute("v18", "player", "eve")
    two_player_graph.add_state_attribute("v19", "player", "eve")

    two_player_graph.add_edge("v0", "v1", weight=0)
    two_player_graph.add_edge("v0", "v2", weight=0)
    two_player_graph.add_edge("v0", "v10", weight=0)
    two_player_graph.add_edge("v10", "v11", weight=0)
    two_player_graph.add_edge("v11", "v10", weight=1)
    two_player_graph.add_edge("v1", "v3", weight=0)
    two_player_graph.add_edge("v1", "v12", weight=0)
    two_player_graph.add_edge("v12", "v1", weight=1)
    two_player_graph.add_edge("v2", "v13", weight=0)
    two_player_graph.add_edge("v13", "v2", weight=1)
    two_player_graph.add_edge("v2", "v4", weight=0)
    
    two_player_graph.add_edge("v3", "v5", weight=0)
    two_player_graph.add_edge("v3", "v6", weight=0)
    two_player_graph.add_edge("v4", "v6", weight=0)
    two_player_graph.add_edge("v6", "v7", weight=0)
    two_player_graph.add_edge("v7", "v8", weight=0)
    two_player_graph.add_edge("v7", "v9", weight=0)
    # leaf node edges
    two_player_graph.add_edge("v5", "v14", weight=3)
    two_player_graph.add_edge("v5", "v15", weight=4)
    two_player_graph.add_edge("v8", "v16", weight=2)
    two_player_graph.add_edge("v8", "v17", weight=9)
    two_player_graph.add_edge("v9", "v18", weight=5)
    two_player_graph.add_edge("v9", "v19", weight=10)

    two_player_graph.add_accepting_states_from(["v14", "v15", "v16", "v17", "v18", "v19"])

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph



def admissibility_game_toy_example_3(plot: bool = False) -> TwoPlayerGraph:
    """
     The example for ICRA 25.
    """

    # build a graph
    two_player_graph = graph_factory.get("TwoPlayerGraph",
                                         graph_name="admissibile_game_3",
                                         config_yaml="/config/admissibile_game_3",
                                         save_flag=True,
                                         from_file=False,
                                         plot=False)

    # circle in this toy example is sys(eve) and square is env(adam)
    two_player_graph.add_states_from(["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", \
                                       "v11", "v12", "v13"])
    
    two_player_graph.add_initial_state('v0')
    two_player_graph.add_state_attribute("v0", "player", "eve")
    two_player_graph.add_state_attribute("v1", "player", "adam")
    two_player_graph.add_state_attribute("v2", "player", "adam")
    two_player_graph.add_state_attribute("v3", "player", "eve")
    two_player_graph.add_state_attribute("v4", "player", "adam")
    two_player_graph.add_state_attribute("v5", "player", "adam")
    two_player_graph.add_state_attribute("v6", "player", "eve")
    two_player_graph.add_state_attribute("v7", "player", "adam")
    two_player_graph.add_state_attribute("v8", "player", "eve")
    two_player_graph.add_state_attribute("v9", "player", "adam")
    two_player_graph.add_state_attribute("v10", "player", "adam")
    two_player_graph.add_state_attribute("v11", "player", "eve")
    two_player_graph.add_state_attribute("v12", "player", "adam")
    two_player_graph.add_state_attribute("v13", "player", "eve")

    # add edges and edge weight
    two_player_graph.add_edge("v0", "v1", weight=1)
    two_player_graph.add_edge("v1", "v0", weight=0)
    two_player_graph.add_edge("v0", "v2", weight=1)
    two_player_graph.add_edge("v2", "v3", weight=0)
    two_player_graph.add_edge("v3", "v2", weight=1)
    two_player_graph.add_edge("v0", "v4", weight=1000)
    two_player_graph.add_edge("v0", "v5", weight=1)
    two_player_graph.add_edge("v5", "v6", weight=0)
    two_player_graph.add_edge("v6", "v7", weight=1)
    two_player_graph.add_edge("v7", "v6", weight=0)
    two_player_graph.add_edge("v4", "v8", weight=0)
    two_player_graph.add_edge("v5", "v8", weight=0)
    two_player_graph.add_edge("v8", "v9", weight=100)
    two_player_graph.add_edge("v9", "v8", weight=0)
    two_player_graph.add_edge("v8", "v10", weight=1)
    two_player_graph.add_edge("v10", "v11", weight=0)
    two_player_graph.add_edge("v11", "v12", weight=1)
    two_player_graph.add_edge("v12", "v11", weight=0)
    two_player_graph.add_edge("v9", "v13", weight=0)
    two_player_graph.add_edge("v10", "v13", weight=0)

    # accepting state self-loop
    two_player_graph.add_edge("v13", "v13", weight=0)

    two_player_graph.add_accepting_states_from(["v13"])

    if plot:
        two_player_graph.plot_graph()

    return two_player_graph


