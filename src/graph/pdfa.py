import networkx as nx
import re
import yaml
import numpy as np

from typing import List, Tuple, Dict
from graphviz import Digraph

# import local packages
from .base import Graph
from ..factory.builder import Builder
from ..spot.promela import parse as parse_ltl, find_states, find_symbols
from ..spot.spot import run_spot
from ..spot.Parser import parse as parse_guard


class PDFAGraph(Graph):

    def __init__(self, graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 use_alias: bool = False):
        self._graph_name = graph_name
        self._use_alias = use_alias
        Graph.__init__(self, config_yaml=config_yaml, save_flag=save_flag)

    def construct_graph(self, plot: bool = False):
        buchi_automaton = nx.MultiDiGraph(name=self._graph_name)
        self._graph = buchi_automaton

        # Read yaml file and load to self._graph_yaml
        self.read_yaml_file()

        self._construct_pdfa(self._graph_yaml)

        if plot:
            self.plot_graph()

    def _construct_pdfa(self, graph_yaml):
        """
        A method that construct a PDFA keeping the original naming convention intact.
        It does not change the name of the nodes we get originally from SPOT
        :param states:
        :param edges:
        # :param initial_states:
        # :param accepting_states:
        :return: An instance of PDFAGraph with nodes, edges, and transition labels that enable those transitions
        """

        # add nodes
        for node_name, attr in graph_yaml['nodes'].items():
            self.add_state(node_name)
            for attr_name, attr_val in attr.items():
                self.add_state_attribute(node_name, attr_name, attr_val)

            # add init and accepting node attribute
            if node_name == graph_yaml['start_state']:
                self.add_initial_state(node_name)

            if attr['final_probability'] > 0:
                self.add_accepting_state(node_name)

        # add edges
        for start_name, edge_dict in graph_yaml['edges'].items():
            for end_name, attr in edge_dict.items():
                symbols = attr['symbols']
                probs = attr['probabilities']

                # transition_formula = ' && '.join([f'({s})' for s in symbols])
                # transition_expr = parse_guard(transition_formula)
                for symbol, prob in zip(symbols, probs):
                    transition_formula = f'({symbol})'
                    transition_expr = parse_guard(transition_formula)

                    self.add_edge(start_name,
                                  end_name,
                                  symbol=symbol,
                                  prob=prob,
                                  weight=float(-np.log(prob)),
                                  guard=transition_expr,
                                  guard_formula=transition_formula)

    def dump_to_yaml(self, config_file_name: str = None) -> None:
        """
        A method to dump the contents of the @self._graph in to @self._file_name yaml document which the Graph()
        class @read_yaml_file() reads to visualize it. By convention we dump files into config/file_name.yaml file.

        A sample dump should looks like this :

        >>> graph :
        >>>    vertices:
        >>>             tuple
        >>>             {'player' : 'eve'/'adam'}
        >>>    edges:
        >>>        parent_node, child_node, edge_weight
        """
        if config_file_name is None:
            config_file_name = str(self._config_yaml + '.yaml')

        config_file_add = Graph._get_project_root_directory() + config_file_name

        data_dict = dict(
                alphabet_size=len(self._graph.edges()),
                num_states=len(self._graph.nodes),
                num_obs=3,
                start_state=self.get_initial_states()[0][0],
                nodes=[node for node in self._graph.nodes.data()],
                edges=[edge for edge in self._graph.edges.data()]
        )

        try:
            with open(config_file_add, 'w') as outfile:
                yaml.dump(data_dict, outfile, default_flow_style=False)

        except FileNotFoundError:
            print(FileNotFoundError)
            print(f"The file {config_file_name} could not be found."
                  f" This could be because I could not find the folder to dump in")

    def read_yaml_file(self, file_name: str = None) -> None:
        """
        Reads the configuration yaml file @self._config_yaml associated with graph of
        type Networkx.LabelledDiGraph and store it in @self._graph_yaml
        :return:
        """
        if self._config_yaml is not None:
            if file_name is None:
                file_name = self._config_yaml + ".yaml"

            file_add = Graph._get_project_root_directory() + file_name
            try:
                with open(file_add, 'r') as stream:
                    graph_data = yaml.load(stream, Loader=yaml.Loader)

            except FileNotFoundError as error:
                print(error)
                print(f"The file {file_name} does not exist")

            self._graph_yaml = graph_data

    def plot_graph(self):
        """
        A helper method to dump the graph data to a yaml file, read the yaml file and plotting the graph itself
        :return: None
        """
        autogen_filename = str(self._config_yaml + '_autogen.yaml')
        # dump to yaml file
        self.dump_to_yaml(autogen_filename)
        # read the yaml file
        self.read_yaml_file(autogen_filename)
        # plot it
        self.fancy_graph()

    def fancy_graph(self, color=("lightgrey", "red", "purple")) -> None:
        dot: Digraph = Digraph(name="graph")
        nodes = self._graph_yaml["nodes"]

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
            dot.edge(f'{str(edge[0])}', f'{str(edge[1])}', label=str(edge[2].get('guard_formula')) + ': ' + str(edge[2].get('prob')) )

        # set graph attributes
        dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, True)

    def _from_spot(self, sc_ltl: str) -> Tuple:
        """
        A method to interpret spot's output and storing it in approproate varibales

        states = A list of states in the automaton
        edges = A list of edges in the automaton
        accepts = A list of accepting states in the automaton
        :param sc_ltl:
        :return: A tuple of states, edges and accepting states
        """

        spot_output = run_spot(formula=sc_ltl)
        edges = parse_ltl(spot_output)
        (states, initial, accepts) = find_states(edges)

        return states, edges, initial, accepts

    # TODO: FIX THIS FUNC
    def get_symbols(self) -> List[str]:
        """
        A method that returns the set symbols that constitute a formula
        :param sc_ltl: The input formula to SPOT based on which we construct an automaton.
         Ideal an ltl formula should be composed of a set of symbols which are observations or atomic propositions
          associated with each states in the Transition system. Generally ap represent the truth value assigned to a
           variable
        :return:
        """
        symbols = find_symbols(self._formula)

        return symbols

    def get_absorbing_states(self) -> List[Tuple]:
        abs_states = []
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 1 and list(self._graph.successors(_n))[0] == _n:
                abs_states.append(_n)

        return abs_states


class PDFABuilder(Builder):
    """
    Implements the generic graph builder class for TwoPlayerGraph
    """
    def __init__(self):
        """
        Constructs a new instance of the TwoPlayerGraph Builder
        """
        Builder.__init__(self)

    def __call__(self, graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 sc_ltl: str = "",
                 use_alias: bool = False,
                 plot: bool = False) -> 'PDFAGraph':

        if not (isinstance(sc_ltl, str) or sc_ltl == ""):
            raise TypeError(f"Please ensure that the ltl formula is of type string and is not empty. \n")

        if use_alias:
            print(f"Using custom names for automaton nodes instead of the original ones by SPOT toolbox")

        self._instance = PDFAGraph(graph_name=graph_name,
                                  config_yaml=config_yaml,
                                  save_flag=save_flag,
                                  use_alias=use_alias)

        self._instance.construct_graph(plot=plot)

        return self._instance
