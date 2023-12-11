import re
import networkx as nx

from collections import defaultdict
from typing import List, Dict
from graphviz import Digraph
from sympy import symbols

from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import ter2symb, simplify_guard
from ..spot.Parser import parse as parse_guard

from ..factory.builder import Builder
from .dfa import DFAGraph
from ..spot.Parser import ANDExpression, ORExpression, NotExpression, SymbolExpression, TrueExpression



class LTLfDFAGraph(DFAGraph):

    def __init__(self,
                 formula: str,
                 graph_name: str,
                 config_yaml: str,
                 verbose: bool = False,
                 save_flag: bool = False):
        """
         A class that call the LTLF2DFA python package and constructs DFA.

         Implementation: LTL2DFA package calls the Mona packages to construct a
            minimal DFA.
            
         Link: https://whitemech.github.io/LTLf2DFA/
         Link: https://www.brics.dk/mona/
            
         Syft is another toolbox that follows a symbolic approach. 
         1) It first converts the LTLf formula into a First-order Formula (FOF)
         2) Then, calls Mona to convert to Convert FOF to a minimal DFA.
        """
        DFAGraph.__init__(self, formula=formula, graph_name=graph_name, config_yaml=config_yaml, save_flag=save_flag, use_alias=False)
        self._mona_dfa: str = ''
        self._task_labels: List[str] = []
        self._init_state: List[str] = []
        self._accp_states: List[str] = []
        self._num_of_states: int = None
        self._transitions: Dict[tuple, SymbolExpression] = {}
        
        # Construct mona DFA
        self.construct_dfa(verbose=verbose)
    

    @property
    def mona_dfa(self) -> str:
        return self._mona_dfa
    
    @property
    def task_labels(self) -> str:
        return self._task_labels
    
    @property
    def accp_states(self) -> str:
        return self._accp_states
    
    @property
    def init_state(self) -> str:
        return self._init_state
    
    @property
    def num_of_states(self) -> str:
        return self._num_of_states
    
    @property
    def transitions(self) -> str:
        return self._transitions
    

    def get_symbols(self) -> List[str]:
        return self.task_labels
    

    def construct_graph(self, debug: bool = False, plot: bool = False, **kwargs):
        """
         The function that parses the mon-dfa output and constructs a graph.
        """
        buchi_automaton = nx.MultiDiGraph(name=self._graph_name)
        self._graph = buchi_automaton
        self.add_states_from([f"q{s}" for s in range(1, self.num_of_states + 1)])
        self.add_initial_state(*self.init_state)
        self.add_accepting_states_from(self.accp_states)

        and_transitions = defaultdict(lambda: [])
        or_transitions: Dict[tuple, str] = {}

        for line in self.mona_dfa.splitlines():
            if line.startswith("State "):
                # extract the original state
                orig_state = self.get_value(line, r".*State[\s]*(\d+):\s.*", int)
                orig_state_ = f"q{orig_state}"
                
                # extract string guard
                guard = self.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                
                # at each iteration with a AND expression which we will store in a list
                if self.task_labels:
                    transition_expr = self.get_ltlf_edge_formula(guard)
                else:
                    transition_expr = self.get_ltlf_edge_formula("X")
                
                dest_state = self.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)
                dest_state_ = f"q{dest_state}"

                # ignore the superficial state 0
                if orig_state:
                    and_transitions[(orig_state_, dest_state_)].append(transition_expr)
                    if debug:
                        print(f" Mona edge: {guard}")
                        print(f"Symbolic AND Edge: {orig_state_} ------{transition_expr.__repr__()}------> {dest_state_}")

        
        for (source, dest), edges in and_transitions.items():
            expr = NotExpression(TrueExpression())
            for sym_edge in edges:
                expr = ORExpression(expr, sym_edge)
            
            # for expression(s) that are of type TrueExpression(), we convert them to SymbolExpression() with name '(True)'.
            # this is done to be consistent with spot's output and to keep the product construction code consistent.
            if and_transitions[(source, dest)][0].name == 'TRUE':
                assert len(and_transitions[(source, dest)]) == 1, "[Error] Constructing DFA. The (true) edge construction in DFA failed. FIX THIS!!!"
                or_transitions[(source, dest)] = parse_guard('(true)')
                self.add_edge(source, dest, guard=parse_guard('(true)'), guard_formula=expr.__repr__())
            else:
                # need to for product construction
                expr.formula = None
                or_transitions[(source, dest)] = expr
                self.add_edge(source, dest, guard=expr, guard_formula=expr.__repr__())

            if debug:
                print(f"Final OR Edge from {source} ------{expr.__repr__()}------> {dest}")
        
        self._transitions = or_transitions

        if plot:
            self.plot_graph(**kwargs)


    def get_ltlf_edge_formula(self, guard: str):
        """
         A function that parse the guard and constructs its correpsonding edge for LTLf DFA construction.
        """

        expr = TrueExpression()

        for idx, value in enumerate(guard):
            if value == "1":
                expr = ANDExpression(expr, self.task_labels[idx])
            elif value == "0":
                expr = ANDExpression(expr, NotExpression(self.task_labels[idx]))
            else:
                assert value == "X", "Error while constructing symbolic LTLF DFA edge. FIX THIS!!!"
        
        return expr
    

    def get_value(self, text, regex, value_type=float):
        """
         Dump a value from a file based on a regex passed in. This function is borrowed from LTLf2DFA/ltlf2dfa/ltlf2dfa.py script
        """
        pattern = re.compile(regex, re.MULTILINE)
        results = pattern.search(text)
        if results:
            return value_type(results.group(1))
        else:
            print("Could not find the value {}, in the text provided".format(regex))
            return value_type(0.0)


    def parse_mona(self, debug: bool = False):
        """
        Parse mona output and extract the initial, accpeting and other states.
          The edges are constructed by get_ltlf_edge_formula(). 
        """
        free_variables = self.get_value(
            self._mona_dfa, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str
        )
        if "state" in free_variables:
            free_variables = None
            task_labels = None
        else:
            task_labels: List[SymbolExpression] = []
            for x in free_variables.split():
                if len(x.strip()) > 0:
                    task_labels.append(SymbolExpression(x.strip().lower()))
        
        # store task specific labels
        self._task_labels = task_labels

        self._init_state = ['q1']
        accepting_states = self.get_value(self._mona_dfa, r".*Accepting states:[\s]*(.*?)\n.*", str)
        accepting_states = [
            str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0
        ]
        # store accepting states
        self._accp_states = [f"q{int(i)}" for i in accepting_states]

        # store # DFA states
        self._num_of_states = self.get_value(self._mona_dfa, '.*Automaton has[\s]*(\d+)[\s]states.*', int) - 1

        # adding print statements to debug
        if debug:
            print("init state: ", self.init_state)
            print("accepting states: ", self.accp_states)
            print("num of states: ", self.num_of_states)
            print("task labels: ", self.task_labels.__repr__())

    
    def construct_dfa(self,
                      verbose: bool = False):
        """
         A helper function that calls Mona and then parse the output and construct a DFA.
        """
        parser = LTLfParser()
        # returns an LTLf Formula
        formula = parser(self._formula)       

        # LTLf to Mona DFA
        mona_dfa = formula.to_dfa(mona_dfa_out=True)
        
        self._mona_dfa = mona_dfa
        
        if verbose:
            print("********************Mona DFA********************")
            print(mona_dfa)  

        
        if "Formula is unsatisfiable" in mona_dfa:
            print("Unsat Formula")
        else:
            my_dfa = self.parse_mona(debug=verbose)
        
        return my_dfa

    def _fancy_edge_labels(self) -> Dict[tuple, str]:
        """
         A function that returns the fancy edge labels for the graph by calling the LTLfDFA toolbox.
           This function basically recreates the transition using LTLF2DFA toolbox's functions.
        """
        and_dot_trans = {}  # maps each couple (src, dst) to a list of guards
        _ap = tuple([symbols(prop.name) for prop in self._task_labels])
        
        for line in self._mona_dfa.splitlines():
            if line.startswith("State "):
                orig_state = self.get_value(line, r".*State[\s]*(\d+):\s.*", int)
                guard = self.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                if _ap:
                    guard = ter2symb(_ap, guard)
                else:
                    guard = ter2symb(_ap, "X")
                dest_state = self.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)
                if orig_state:
                    if (f"q{orig_state}", f"q{dest_state}") in and_dot_trans:
                        and_dot_trans[(f"q{orig_state}", f"q{dest_state}")].append(guard)
                    else:
                        and_dot_trans[(f"q{orig_state}", f"q{dest_state}")] = [guard]

        or_dot_trans = {}
        for c, guards in and_dot_trans.items():
            simplified_guard = simplify_guard(guards)
            or_dot_trans[(c[0], c[1])] = str(simplified_guard).lower()
        
        return or_dot_trans

    def fancy_graph(self, color=("lightgrey", "red", "purple"), **kwargs) -> None:
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


        # for guard formula we call the original ltlf2dfa tool as the formulas are more readable
        fancy_edges = self._fancy_edge_labels()

        for (src, dest), edge in fancy_edges.items():
            dot.edge(f'{src}', f'{dest}', label=edge)

        # set graph attributes
        dot.graph_attr['rankdir'] = 'LR'
        dot.node_attr['fixedsize'] = 'False'
        dot.edge_attr.update(arrowhead='vee', arrowsize='1', decorate='True')

        if self._save_flag:
            graph_name = str(self._graph.__getattribute__('name'))
            self.save_dot_graph(dot, graph_name, **kwargs)


class LTLfDFABuilder(Builder):

    """
     Implements the generic graph builder class for LTLf DFA Construction from LTLF2DFA python package
    """
    def __init__(self):
        """
         Constructs a new instance of the LTLfDFAGraph Builder
        """
        Builder.__init__(self)

    def __call__(self, graph_name: str,
                 config_yaml: str,
                 save_flag: bool = False,
                 ltlf: str = "",
                 plot: bool = False,
                 view: bool = True,
                 format: str = 'png',
                 ) -> 'LTLfDFAGraph':

        if not (isinstance(ltlf, str) or ltlf == ""):
            raise TypeError(f"Please ensure that the ltl formula is of type string and is not {type(ltlf)}. \n")

        self._instance = LTLfDFAGraph(formula=ltlf,
                                      graph_name=graph_name,
                                      config_yaml=config_yaml,
                                      save_flag=save_flag)

        self._instance.construct_graph(plot=plot, view=view, format=format)

        return self._instance


if __name__ == "__main__":
    formula = 'F(a & F(b))'

    dfa_handle = LTLfDFAGraph(formula=formula)
    dfa_handle.construct_graph()
