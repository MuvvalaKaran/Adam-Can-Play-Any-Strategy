from graphviz import Digraph

from src.graph import FiniteTransSys
from src.factory.builder import Builder


class MiniGrid(FiniteTransSys):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        FiniteTransSys.__init__(self, graph_name, config_yaml, save_flag)

    def construct_graph(self):
        super().construct_graph()

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        """
               Method to create a illustration of the graph
               :return: Diagram of the graph
               """
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]
        for n in nodes:
            ap = n[1].get('xlabel')
            color = n[1].get('color')
            dot.node(n[0], _attributes={"style": "filled", "fillcolor": color, "xlabel": ap, "shape": "rectangle"})

        # add all the edges
        edges = self._graph_yaml["edges"]

        # load the weights to illustrate on the graph
        for counter, edge in enumerate(edges):
            label = edge[2].get('label')
            fontcolor = edge[2].get('fontcolor')
            dot.edge(str(edge[0]), str(edge[1]), label=label, fontcolor=fontcolor)

        # set graph attributes
        # dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)


class MiniGridBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self,
                 graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 plot: bool = False) -> 'MiniGrid':

        """
        A function to build the TS from the gym-minigrid env from a config_file.

        NOTE: For now we only have the provision to build this TS from a yaml file
        :param graph_name:
        :param config_yaml:
        :param save_flag:
        :param plot:
        :return:
        """

        self._instance = MiniGrid(graph_name, config_yaml, save_flag=save_flag)
        self._instance.construct_graph()
        self._instance._graph_yaml = self._from_yaml(config_yaml)

        if plot:
            self._instance.fancy_graph()

        return self._instance

    def _from_yaml(self, config_file_name: str) -> dict:

        config_data = self.load_YAML_config_data(config_file_name)

        return config_data