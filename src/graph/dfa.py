import networkx as nx
import re

from typing import List, Tuple, Dict
from graphviz import Digraph

# import local packages
from .base import Graph
from src.factory.builder import Builder
from src.spot.promela import parse as parse_ltl, find_states, find_symbols
from src.spot.spot import run_spot
from src.spot.Parser import parse as parse_guard


class DFAGraph(Graph):

    def __init__(self, formula: str,
                 graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 use_alias: bool = False):
        # initialize the Graph class instance variables
        self._formula = formula
        self._graph_name = graph_name
        self._use_alias = use_alias
        Graph.__init__(self, config_yaml=config_yaml, save_flag=save_flag)

    def construct_graph(self, plot: bool = False):
        buchi = nx.MultiDiGraph(name=self._graph_name)
        self._graph = buchi

        states, edges, initial, accepts = self._from_spot(self._formula)
        
        if self._use_alias:
            self._construct_dfa_w_alias(states, edges)
        else:
            self._construct_dfa(states, edges)

        if plot:
            self.plot_graph()
    
    def _construct_dfa(self, states, edges):
        """
        A method that construct a DFA keeping the original naming convention intact. 
        It does not change the name of the nodes we get originally from SPOT
        :param states: 
        :param edges: 
        # :param initial_states: 
        # :param accepting_states: 
        :return: An instance of DFAGraph with nodes, edges, and transition labels that enable those transitions
        """
        # add nodes
        for _s in states:
            self.add_state(_s)
            
            # add init and accepting node attribute 
            if _s == "T0_init":
                self.add_initial_state(_s)
            
            if _s == "accept_all":
                self.add_accepting_state(_s)
        
        # add edges
        for (u, v) in edges.keys():
            transition_formula = edges[(u, v)]
            transition_expr = parse_guard(transition_formula)
            self.add_edge(u,
                          v,
                          guard=transition_expr,
                          guard_formula=transition_formula)
    
    def _construct_dfa_w_alias(self, states, edges):
        """
        A method that construct a DFA but changes the original node names
        :param states:
        :param edges:
        :return: An instance of DFAGraph with nodes, edges, and transition labels that enable those transitions
        """
        # convert nodes from their original names to personal naming convention
        _new_states_map = self._convert_std_state_names(states)

        # add nodes
        for _, _s in _new_states_map.items():
            self.add_state(_s)

            if _s == "q1":
                self.add_initial_state(_s)

            if _s == "q0":
                self.add_accepting_state(_s)

        # add edges
        for (u, v) in edges.keys():
            transition_formula = edges[(u, v)]
            transition_expr = parse_guard(transition_formula)
            self.add_edge(_new_states_map[u],
                          _new_states_map[v],
                          guard=transition_expr,
                          guard_formula=transition_formula)

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

    def _convert_std_state_names(self, states: List[str]) -> Dict[str, str]:
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

    def _get_symbols(self) -> List[str]:
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

    def _get_absorbing_states(self) -> List[Tuple]:
        abs_states = []
        for _n in self._graph.nodes():
            if len(list(self._graph.successors(_n))) == 1 and list(self._graph.successors(_n))[0] == _n:
                abs_states.append(_n)

        return abs_states


class DFABuilder(Builder):
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
                 plot: bool = False) -> 'DFAGraph':

        if not (isinstance(sc_ltl, str) or sc_ltl == ""):
            raise TypeError(f"Please ensure that the ltl formula is of type string and is not empty. \n")

        if use_alias:
            print(f"Using custom names for automaton nodes instead of the original ones by SPOT toolbox")

        self._instance = DFAGraph(formula=sc_ltl,
                                  graph_name=graph_name,
                                  config_yaml=config_yaml,
                                  save_flag=save_flag,
                                  use_alias=use_alias)

        self._instance.construct_graph(plot=plot)

        return self._instance
