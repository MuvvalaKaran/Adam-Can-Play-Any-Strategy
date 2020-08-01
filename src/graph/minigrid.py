import warnings

from graphviz import Digraph

from src.graph import FiniteTransSys
from src.factory.builder import Builder


class MiniGrid(FiniteTransSys):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        FiniteTransSys.__init__(self, graph_name, config_yaml, save_flag)

    def build_graph_from_file(self):
        """
        A method to build the graph from a config file. Before we run this method we need to make sure that the
        graph data like nodes and edges and their respective attributes have been store in a self._graph_yaml attribute.
        :return: updates the graph with the respective nodes and the edges
        """

        if self._graph_yaml is None:
            warnings.warn("Please ensure that you have first loaded the config data. You can do this by"
                          "setting the respective True in the builder instance.")

        _nodes = self._graph_yaml['nodes']
        _start_state = self._graph_yaml['start_state']

        # each node has an atomic proposition and a player associated with it. Some states also init and
        # accepting attributes associated with them
        for _n in _nodes:
            state_name = _n[0]
            ap = _n[1].get('observation')
            # all nodes we get from the gym minigrid be default belong to system/eve
            # player = _n[1].get('player')
            self.add_state(state_name, ap=ap, player="eve")

            if _n[1].get('is_accepting'):
                self.add_accepting_state(state_name)

        self.add_initial_state(_start_state)

        _edges = self._graph_yaml['edges']

        # as originally the minigrid world does not have weight associated with its actions,
        # we will manually assign a weight here

        ACTION_STR_TO_WT = {
            'north': -1,
            'south': -2,
            'east': -3,
            'west': -4
        }

        # NOTE : ALL actions have the same cost of 1 unless specified in the yaml specifically
        for _e in _edges:
            _weight = _e[2].get('weight')
            _action = _e[2].get('label')

            if _weight is None:
                self.add_edge(_e[0], _e[1], weight=ACTION_STR_TO_WT[_action], actions=_action)
            else:
                self.add_edge(_e[0], _e[1], weight=_weight, actions=_action)

    def fancy_graph(self, color=()) -> None:
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