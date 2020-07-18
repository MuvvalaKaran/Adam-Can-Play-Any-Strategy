from .two_player_graph import TwoPlayerGraph
from src.factory.builder import Builder

class FiniteTransSys(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._graph_name = graph_name
        self._config_yaml = config_yaml
        self._save_flag = save_flag

    def construct_graph(self):
        super().construct_graph()

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
        Method to create a illustration of the graph
        :return: Diagram of the graph
        """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["vertices"]
        for n in nodes:
            # default color for all the nodes is grey
            ap = n[1].get('ap')
            ap = "{" + str(ap) + "}"
            dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[0], "xlabel": ap})
            if n[1].get('init'):
                # default color for init node is red
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[1], "xlabel": ap})
            if n[1].get('accepting'):
                # default color for accepting node is purple
                dot.node(str(n[0]), _attributes={"style": "filled", "fillcolor": color[2], "xlabel": ap})
            if n[1].get('player') == 'eve':
                dot.node(str(n[0]), _attributes={"shape": "rectangle", "xlabel": ap})
            else:
                dot.node(str(n[0]), _attributes={"shape": "circle", "xlabel": ap})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            # ap_u = self._graph.nodes[edge[0]].get('ap')
            # ap_v = self._graph.nodes[edge[1]].get('ap')
            if edge[2].get('strategy') is True:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')),
                         _attributes={'color': 'red'})
            else:
                dot.edge(str(edge[0]), str(edge[1]), label=str(edge[2].get('actions')))

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    def automate_construction(self, k: int):
        # a function to construct the two game automatically in code. K = # of times the human can intervene
        if not isinstance(k, int):
            warnings.warn("Please Make sure the Quantity K which represents the number of times the human can "
                          "intervene is an integer")
        eve_node_lst = []
        adam_node_lst = []
        two_player_graph_ts = FiniteTransSys("transition_system", self._config_yaml, self._save_flag)
        two_player_graph_ts.construct_graph()

        # lets create k copies of the stats
        for _n in self._graph.nodes():
            for i in range(k+1):
                _sys_node = (_n, i)
                eve_node_lst.append(_sys_node)

        two_player_graph_ts.add_states_from(eve_node_lst, player='eve')

        # for each edge create a human node and then alter the orginal edge to go through the human node
        for e in self._graph.edges():
            for i in range(k):
                # lets create a human edge with huv,k naming convention
                _env_node = ((f"h{e[0][1:]}{e[1][1:]}"), f"{i}")
                adam_node_lst.append(_env_node)

        two_player_graph_ts.add_states_from(adam_node_lst, player='adam')

        # add init node
        init_node = self.get_initial_states()
        two_player_graph_ts.add_state_attribute((init_node[0][0], 0), "init", True)

        # now we add edges
        for e in self._graph.edges.data():
            # add edge between e[0] and the human node h{e[0][1:]}{e[0][1:]}, k
            for ik in range(k):
                two_player_graph_ts.add_edge((e[0], ik), ((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"),
                                             actions=e[2].get("actions"), weight=e[2].get("weight"))
                two_player_graph_ts.add_edge(((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"), (e[1], ik),
                                             actions=e[2].get("actions"), weight=e[2].get("weight"))
                _alt_nodes_set = set(self._graph.nodes()) - {e[1]}
                for _alt_nodes in _alt_nodes_set:
                    two_player_graph_ts.add_edge(((f"h{e[0][1:]}{e[1][1:]}"), f"{ik}"), (_alt_nodes, ik+1),
                                                 actions="m", weight="0")

        # manually add edges to states that belong to k index
        for e in self._graph.edges.data():
            two_player_graph_ts.add_edge((e[0], k), (e[1], k),
                                         actions=e[2].get('actions'), weight=e[2].get("weight"))

        # add the original atomic proposition to the new states
        for _n in self._graph.nodes.data():
            if _n[1].get('ap'):
                for ik in range(k+1):
                    two_player_graph_ts.add_state_attribute((_n[0], ik), 'ap', _n[1].get('ap'))

        return two_player_graph_ts

    def get_max_weight(self) -> str:
        # NOTE: WE assuming that the edge weights are purely integer

        max_weight: int = -1
        # loop through all the edges and return the max weight
        for _e in self._graph.edges.data("weight"):
            if int(_e[2]) > max_weight:
                max_weight = int(_e[2])

        return str(max_weight)


class TransitionSystemBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self, **kwargs):
        pass