from src.graph import graph

if __name__ == "__main__":
    g = graph.get("TwoPlayerGraph", graph_name="org_grpah", config_yaml="config/org_graph", save_flag=True, pre_built=True)
    g.plot_graph()