import networkx as nx
import re


from .base import Graph
from src.factory.builder import Builder

from typing import List, Tuple, Dict
from graphviz import Digraph


class DFAGraph(Graph):

    def __init__(self, formula: str, graph_name: str, config_yaml: str, save_flag: bool = False):
        # initialize the Graph class instance variables
        self._formula = formula
        self._config_yaml = config_yaml
        self._save_flag = save_flag
        self._graph_name = graph_name
        # self._absorbing_states = self.get_absorbing_states()

    def construct_graph(self):
        buchi = nx.MultiDiGraph(name=self._graph_name)
        self._graph = buchi

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["vertices"]

        for n in nodes:
            # default color for all the nodes is grey
            dot.node(f'{str(n[0])}', _attributes={"shape": "circle", "style": "filled", "fillcolor": color[0]})
            if n[1].get("init"):
                # default color for init node is red
                dot.node(f'{str(n[0])}', _attributes={"style": "filled", "fillcolor": color[1]})
            if n[1].get("accepting"):
                # default color for accepting node is purple
                dot.node(f'{str(n[0])}',
                         _attributes={"shape": "doublecircle", "style": "filled", "fillcolor": color[2]})

        # add all the edges
        edges = self._graph_yaml["edges"]

        for counter, edge in enumerate(edges):
            dot.edge(f'{str(edge[0])}', f'{str(edge[1])}', label=str(edge[2].get('guard_formula')))

        # set graph attributes
        dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    def convert_std_state_names(self, states: List[str]) -> Dict[str, str]:
        """
        A helper function to change the name of a state of the format
        1. T0_S# -> q#
        2. T0_init -> qW where W is the max number of states
        3. accept_all -> q0
        :param states: A List of states with the original naming convention of spot
        :return: A list of state with the new naming convention
        """
        _new_state_lst = {}
        for _s in states:
            if _s == "T0_init":
                _new_state_lst.update({_s: f"q1"})
            elif _s == "accept_all":
                _new_state_lst.update({_s: "q0"})
            else:
                # find the number after the string T0_S
                s = re.compile("^T0_S")
                indx = s.search(_s)
                # number string is
                _new_state_lst.update({_s: f"q{int(_s[indx.regs[0][1]:])}"})

        return _new_state_lst

    def get_absorbing_states(self) -> List[Tuple]:
        abs_states = []
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 1 and list(self._graph.successors(_n))[0] == _n:
                abs_states.append(_n)

        return abs_states


class DFABuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self, **kwargs):
        pass