import re
import networkx as nx

from collections import defaultdict
from typing import List
from sympy import symbols, And, Not, Or, simplify

from ..spot.Parser import ANDExpression, ORExpression, NotExpression, SymbolExpression, TrueExpression
from ltlf2dfa.parser.ltlf import LTLfParser
from .dfa import DFAGraph


class LTLfDFAGraph(DFAGraph):

    def __init__(self,
                 formula: str,
                 graph_name: str,
                 config_yaml: str,
                 verbose: bool = False,
                 save_flag: bool = False,
                 use_alias: bool = False):
        """
         A class that call the LTLF2DFA python package and constructs DFA.

         Implementation: LTL2DFA package calls the Mona packages to construct a
            minimal DFA.
            
         Link: https://whitemech.github.io/LTLf2DFA/
         Link: https://www.brics.dk/mona/
            
         Syft is another toolbox that follows a symbolic approach. 
         1) It first converts the LTLf fomrula into a First-order Formula (FOF)
         2) Then, calls Mona to convert to Convert FOF to a minimal DFA.
         3) Finally, Syft uses BDD based symbolic representation to construct a 
        """
        DFAGraph.__init__(self, formula=formula, graph_name=graph_name, use_alias=use_alias, config_yaml=config_yaml, save_flag=save_flag)
        self.mona_dfa: str = ''
        self.task_labels: list = []
        self.init_state: list = []
        self.accp_states: list = []
        self.num_of_states = None
        
        # Construct mona DFA
        self.construct_dfa(verbose=verbose)

    def construct_graph(self, plot: bool = False, **kwargs):
        """
         The function that parses the mon-dfa output and constructs a graph.
        """
        # create graph
        buchi_automaton = nx.MultiDiGraph(name=self._graph_name)
        self._graph = buchi_automaton
        self.add_states_from([f"q{s}" for s in range(1, self.num_of_states + 1)])
        self.add_initial_state(*self.init_state)
        self.add_accepting_states_from(self.accp_states)

        transitions = defaultdict(lambda: [])

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

                transitions[(orig_state_, dest_state_)].append(transition_expr)
            
                # ignore the superficial state 0
                if orig_state:
                    # assert self._graph.has_edge(orig_state_, dest_state_ ) is False, \
                    # f"[Error] Constructing LTLf DFA. Edge from {orig_state_} ------{self._graph[orig_state_][dest_state_][0]['guard']}------> {dest_state_} exists."\
                    # f"Trying to add {orig_state_} ------{self.simplify_guard(transition_expr)}------> {dest_state_}. FIX THIS!!!"
                    # self.add_edge(orig_state_, dest_state_, guard=transition_expr, guard_formula=str(self.simplify_guard(transition_expr)))
                    print(f" Mona edge: {guard}")
                    print(f"Symbolic Edge: {orig_state_} ------{transition_expr.__repr__()}------> {dest_state_}")
        
        print("Done Constructing DFA")


    def get_ltlf_edge_formula(self, guard: str):
        """
        A function that parse the guard and constructs its correpsonding edge for LTLf DFA construction.
          This function is borrowed from LTLf2DFA/ltlf2dfa/ltlf2dfa.py script 
        """
        # expr = And()
        # expr = ANDExpression

        expr = TrueExpression()

        # print(guard)

        for idx, value in enumerate(guard):
            if value == "1":
                # expr = And(expr, self.task_labels[idx] if isinstance(self.task_labels, tuple) else str(self.task_labels))
                expr = ANDExpression(expr, self.task_labels[idx])
            elif value == "0":
                # expr = And(expr, str(self.task_labels[idx] if isinstance(self.task_labels, tuple) else str(self.task_labels)))
                expr = ANDExpression(expr, NotExpression(self.task_labels[idx]))
            else:
                assert value == "X", "Error while constructing symbolic LTLF DFA edge. FIX THIS!!!"
        
        return expr

    def simplify_guard(self, guards):
        """
        Make a big OR among guards and simplify them. This function is borrowed from LTLf2DFA/ltlf2dfa/ltlf2dfa.py script
        """
        return simplify(Or(guards))
    

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


    def parse_mona(self, mona_output):
        """
        Parse mona output and extract the initial, accpeting and other states.
          The edges are constructed by get_ltlf_edge_formula(). 
        """
        free_variables = self.get_value(
            mona_output, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str
        )
        if "state" in free_variables:
            free_variables = None
        else:
            task_labels: List[SymbolExpression] = []
            for x in free_variables.split():
                if len(x.strip()) > 0:
                    task_labels.append(SymbolExpression(x.strip().lower()))
            # free_variables = SymbolExpression(
            #     " ".join(
            #         x.strip().lower() for x in free_variables.split() if len(x.strip()) > 0
            #     )
            # )
        
        # store task specific labels
        self.task_labels = task_labels

        self.init_state = ['q1']
        accepting_states = self.get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)
        accepting_states = [
            str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0
        ]
        # store accepting states
        self.accp_states = [f"q{int(i)}" for i in accepting_states]

        # store # DFA states
        self.num_of_states = self.get_value(mona_output, '.*Automaton has[\s]*(\d+)[\s]states.*', int) - 1

        # adding print statements to debug
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
        formula = parser(self._formula)       # returns an LTLf Formula

        # LTLf to Mona DFA
        mona_dfa = formula.to_dfa(mona_dfa_out=True)
        
        self.mona_dfa = mona_dfa
        
        if verbose:
            print("********************Mona DFA********************")
            print(mona_dfa)  

        
        if "Formula is unsatisfiable" in mona_dfa:
            print("Unsat Formula")
        else:
            my_dfa = self.parse_mona(mona_dfa)
        
        return my_dfa


if __name__ == "__main__":
    formula = 'F(a & F(b))'

    dfa_handle = LTLfDFAGraph(formula=formula)
    dfa_handle.construct_graph()