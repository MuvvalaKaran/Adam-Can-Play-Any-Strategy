"""
 This file tests the LTL to DFA construction using SPOT. We will test on multiple formulas of different types.

 Things we verify for 

 1. Pasrsing the SPOT Output from the SPOT toolbox (https://gitlab.lre.epita.fr/spot/spot)
 2. DFA graph construction
   2.1 Asserting correct # of states, init states, accepting states, and absorbing states, symbols (atomic propositions) in the constructed DFA
   2.2 Asserting correct # of edges in DFA
   2.3 Asserting correct symbols that enable each edge in the DFA
 3. Checking for plotting inside Docker - we will not plot if we are inside Docker
 4. Checking for plotting outside Docker - we will plot if we are outside Docker
 5. Checking for constructing DFA with alias names
"""
import os
import unittest

from src.spot.Parser import (
    SymbolExpression,
    ORExpression,
    ANDExpression,
    TrueExpression,
    NotSymbolExpression,
)
from src.graph import graph_factory

CURRENT_DIR = os.path.abspath(__file__)
PLOT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ltl_DFA_plots")

FORMULAS = {
    "a": {
        "init_state": ["T0_init"],
        "accp_state": ["accept_all"],
        "absorbing_states": ["accept_all", "T0_S2"],
        "num_of_states": 3,
        "task_labels": ["a"],
        "transitions": [
            (
                "T0_S2",
                "T0_S2",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            ),
            (
                "T0_init",
                "accept_all",
                {"guard": SymbolExpression("a"), "guard_formula": "(a)"},
            ),
            (
                "T0_init",
                "T0_S2",
                {"guard": NotSymbolExpression("a"), "guard_formula": "(!(a))"},
            ),
            (
                "accept_all",
                "accept_all",
                {"guard": TrueExpression(), "guard_formula": "1"},
            ),
        ],
    },
    "true": {
        "init_state": ["accept_init"],
        "accp_state": ["accept_init"],
        "absorbing_states": ["accept_init"],
        "num_of_states": 1,
        "task_labels": ["true"],
        "transitions": [
            (
                "accept_init",
                "accept_init",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            )
        ],
    },
    "F(a & b)": {
        "init_state": ["T0_init"],
        "accp_state": ["accept_all"],
        "absorbing_states": ["accept_all"],
        "num_of_states": 2,
        "task_labels": ["a", "b"],
        "transitions": [
            (
                "T0_init",
                "accept_all",
                {
                    "guard": ANDExpression(
                        SymbolExpression("a"), SymbolExpression("b")
                    ),
                    "guard_formula": "((a) && (b))",
                },
            ),
            (
                "T0_init",
                "T0_init",
                {
                    "guard": ORExpression(
                        NotSymbolExpression("a"), NotSymbolExpression("b")
                    ),
                    "guard_formula": "((!(a)) || (!(b)))",
                },
            ),
            (
                "accept_all",
                "accept_all",
                {"guard": TrueExpression(), "guard_formula": "1"},
            ),
        ],
    },
    "X(a)": {
        "init_state": ["T0_init"],
        "accp_state": ["accept_all"],
        "absorbing_states": ["accept_all", "T0_S3"],
        "num_of_states": 4,
        "task_labels": ["a"],
        "transitions": [
            (
                "T0_S0",
                "accept_all",
                {"guard": SymbolExpression("a"), "guard_formula": "(a)"},
            ),
            (
                "T0_S0",
                "T0_S3",
                {"guard": NotSymbolExpression("a"), "guard_formula": "(!(a))"},
            ),
            (
                "T0_S3",
                "T0_S3",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            ),
            (
                "T0_init",
                "T0_S0",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            ),
            (
                "accept_all",
                "accept_all",
                {"guard": TrueExpression(), "guard_formula": "1"},
            ),
        ],
    },
    "!a U b": {
        "init_state": ["T0_init"],
        "accp_state": ["accept_all"],
        "absorbing_states": ["accept_all", "T0_S2"],
        "num_of_states": 3,
        "task_labels": ["a", "b"],
        "transitions": [
            (
                "T0_S2",
                "T0_S2",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            ),
            (
                "T0_init",
                "accept_all",
                {"guard": SymbolExpression("b"), "guard_formula": "(b)"},
            ),
            (
                "T0_init",
                "T0_init",
                {
                    "guard": ANDExpression(
                        NotSymbolExpression("a"), NotSymbolExpression("b")
                    ),
                    "guard_formula": "((!(a)) && (!(b)))",
                },
            ),
            (
                "T0_init",
                "T0_S2",
                {
                    "guard": ANDExpression(
                        SymbolExpression("a"), NotSymbolExpression("b")
                    ),
                    "guard_formula": "((a) && (!(b)))",
                },
            ),
            (
                "accept_all",
                "accept_all",
                {"guard": TrueExpression(), "guard_formula": "1"},
            ),
        ],
    },
    "G(!a) & F(b)": {
        "init_state": ["T0_init"],
        "accp_state": ["accept_S0"],
        "absorbing_states": ["T0_S2"],
        "num_of_states": 3,
        "task_labels": ["a", "b"],
        "transitions": [
            (
                "T0_S2",
                "T0_S2",
                {"guard": SymbolExpression("true"), "guard_formula": "(true)"},
            ),
            (
                "T0_init",
                "accept_S0",
                {
                    "guard": ANDExpression(
                        NotSymbolExpression("a"), SymbolExpression("b")
                    ),
                    "guard_formula": "((!(a)) && (b))",
                },
            ),
            (
                "T0_init",
                "T0_init",
                {
                    "guard": ANDExpression(
                        NotSymbolExpression("a"), NotSymbolExpression("b")
                    ),
                    "guard_formula": "((!(a)) && (!(b)))",
                },
            ),
            (
                "T0_init",
                "T0_S2",
                {"guard": SymbolExpression("a"), "guard_formula": "(a)"},
            ),
            (
                "accept_S0",
                "accept_S0",
                {"guard": NotSymbolExpression("a"), "guard_formula": "(!(a))"},
            ),
            (
                "accept_S0",
                "T0_S2",
                {"guard": SymbolExpression("a"), "guard_formula": "(a)"},
            ),
        ],
    },
}


class TestParsingMona(unittest.TestCase):
    def create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def construct_DFA(self, formula: str, plot: bool = False, use_alias: bool = False):
        """
        A helper function that call builder to construct the correspond DFA for the LTLf formula
        """
        dfa_handle = graph_factory.get(
            "DFA",
            graph_name="ltl_automaton",
            config_yaml="config/ltl_automaton",
            save_flag=True,
            sc_ltl=formula,
            use_alias=use_alias,
            plot=plot,
            directory=PLOT_DIR,
            filename=f"{formula}_formula",
        )
        return dfa_handle

    def test_dfa_construction(self):
        """
        A function that calls the LTLf DFA Builder and checks for the DFA constructed.
        """
        for formula, exp_op in FORMULAS.items():
            print("DFA Construction for LTL Formula: ", formula)
            dfa_handle = self.construct_DFA(formula=formula)
            self.assertSetEqual(
                set(exp_op["init_state"]), set([dfa_handle.get_initial_states()[0][0]])
            )
            self.assertSetEqual(
                set(exp_op["accp_state"]), set(dfa_handle.get_accepting_states())
            )
            self.assertSetEqual(
                set(exp_op["task_labels"]), set(dfa_handle.get_symbols())
            )
            self.assertSetEqual(
                set(exp_op["absorbing_states"]), set(dfa_handle.get_absorbing_states())
            )
            self.assertEqual(exp_op["num_of_states"], len(dfa_handle.get_states()))

            # for # of edges
            self.assertEqual(
                len(exp_op["transitions"]), len(dfa_handle.get_transitions())
            )
            self.assertListEqual

            # # assert the edges symbols are corrects
            for source, dest, edge_data in exp_op["transitions"]:
                self.assertEqual(
                    edge_data["guard"].__repr__(),
                    dfa_handle.get_edge_attributes(source, dest, "guard").__repr__(),
                )

                self.assertEqual(
                    edge_data["guard_formula"],
                    dfa_handle.get_edge_attributes(source, dest, "guard_formula"),
                )

    def test_dfa_construction_alias(self):
        """
        A function that calls the LTLf DFA Builder and checks for the DFA constructed with alias names.
        """
        for formula, _ in FORMULAS.items():
            print("DFA Construction w alias for LTL Formula: : ", formula)
            dfa_handle = self.construct_DFA(formula=formula, plot=False, use_alias=True)
            if formula == "true":
                self.assertEqual("q01", dfa_handle.get_accepting_states()[0])
                self.assertEqual("q01", dfa_handle.get_initial_states()[0][0])
            else:
                self.assertEqual("q1", dfa_handle.get_initial_states()[0][0])
                self.assertEqual("q0", dfa_handle.get_accepting_states()[0])

    def test_dfa_plotting(self):
        """
        A function that calls the LTLf DFA Builder with the plot flag set to true.
          If we are inside Docker, we will not open the plot. If we are outside Docker, we will plot and display the plot too.
        """
        print(PLOT_DIR)
        self.create_directory(PLOT_DIR)

        for formula, _ in FORMULAS.items():
            print("DFA Plotting for LTL Formula: ", formula)
            self.construct_DFA(formula=formula, plot=True)


if __name__ == "__main__":
    unittest.main()
