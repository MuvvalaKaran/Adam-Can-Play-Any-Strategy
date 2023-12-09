"""
 This file tests the LTLf to DFA construction. We will test on multiple formulas of different types.

 Things we verify for 

 1. Pasrsing the Mona Output from Mona toolbox
 2. DFA graph construction
   2.1 Asserting correct # of states, init states, accepting states, and absoarbing states, symbols (atomic propositions) in DFA
   2.2 Asserting correct # of edges in DFA
   2.3 Asserting correct symbols that enable each edge
"""

import os
import unittest

from src.spot.Parser import (
    SymbolExpression,
    ORExpression,
    ANDExpression,
    NotExpression,
    TrueExpression,
)
from src.graph import graph_factory

FORMULAS = {
    "a": {
        "init_state": ["q1"],
        "accp_state": ["q3"],
        "num_of_states": 3,
        "task_labels": [SymbolExpression("a")],
        "transitions": {
            ("q1", "q2"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), NotExpression(SymbolExpression("a"))),
            ),
            ("q1", "q3"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), SymbolExpression("a")),
            ),
            ("q2", "q2"): SymbolExpression("true"),
            ("q3", "q3"): SymbolExpression("true"),
        },
    },
    "true": {
        "init_state": ["q1"],
        "accp_state": ["q1"],
        "num_of_states": 1,
        "task_labels": None,
        "transitions": {("q1", "q1"): SymbolExpression("true")},
    },
    "F(a & b)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2,
        "task_labels": [SymbolExpression("a"), SymbolExpression("b")],
        "transitions": {
            ("q1", "q1"): ORExpression(
                ORExpression(
                    NotExpression(TrueExpression()),
                    ANDExpression(
                        TrueExpression(), NotExpression(SymbolExpression("a"))
                    ),
                ),
                ANDExpression(
                    ANDExpression(TrueExpression(), SymbolExpression("a")),
                    NotExpression(SymbolExpression("b")),
                ),
            ),
            ("q1", "q2"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(
                    ANDExpression(TrueExpression(), SymbolExpression("a")),
                    SymbolExpression("b"),
                ),
            ),
            ("q2", "q2"): SymbolExpression("true"),
        },
    },
    "X(a)": {
        "init_state": ["q1"],
        "accp_state": ["q4"],
        "num_of_states": 4,
        "task_labels": [SymbolExpression("a")],
        "transitions": {
            ("q1", "q2"): SymbolExpression("true"),
            ("q2", "q3"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), NotExpression(SymbolExpression("a"))),
            ),
            ("q2", "q4"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), SymbolExpression("a")),
            ),
            ("q3", "q3"): SymbolExpression("true"),
            ("q4", "q4"): SymbolExpression("true"),
        },
    },
    "!a U b": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 3,
        "task_labels": [SymbolExpression("a"), SymbolExpression("b")],
        "transitions": {
            ("q1", "q1"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(
                    ANDExpression(
                        TrueExpression(), NotExpression(SymbolExpression("a"))
                    ),
                    NotExpression(SymbolExpression("b")),
                ),
            ),
            ("q1", "q2"): ORExpression(
                ORExpression(
                    NotExpression(TrueExpression()),
                    ANDExpression(
                        ANDExpression(
                            TrueExpression(), NotExpression(SymbolExpression("a"))
                        ),
                        SymbolExpression("b"),
                    ),
                ),
                ANDExpression(
                    ANDExpression(TrueExpression(), SymbolExpression("a")),
                    SymbolExpression("b"),
                ),
            ),
            ("q1", "q3"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(
                    ANDExpression(TrueExpression(), SymbolExpression("a")),
                    NotExpression(SymbolExpression("b")),
                ),
            ),
            ("q2", "q2"): SymbolExpression("true"),
            ("q3", "q3"): SymbolExpression("true"),
        },
    },
    "G(!a) & F(b)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 3,
        "task_labels": [SymbolExpression("a"), SymbolExpression("b")],
        "transitions": {
            ("q1", "q1"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(
                    ANDExpression(
                        TrueExpression(), NotExpression(SymbolExpression("a"))
                    ),
                    NotExpression(SymbolExpression("b")),
                ),
            ),
            ("q1", "q2"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(
                    ANDExpression(
                        TrueExpression(), NotExpression(SymbolExpression("a"))
                    ),
                    SymbolExpression("b"),
                ),
            ),
            ("q1", "q3"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), SymbolExpression("a")),
            ),
            ("q2", "q2"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), NotExpression(SymbolExpression("a"))),
            ),
            ("q2", "q3"): ORExpression(
                NotExpression(TrueExpression()),
                ANDExpression(TrueExpression(), SymbolExpression("a")),
            ),
            ("q3", "q3"): SymbolExpression("true"),
        },
    },
}


class TestParsingMona(unittest.TestCase):
    def construct_DFA(self, formula: str):
        """
        A helper function that call builder to construct the correspond DFA for the LTLf formula
        """
        dfa_handle = graph_factory.get(
            "LTLfDFA",
            graph_name="ltlf_automaton",
            config_yaml="config/ltltf_automaton",
            save_flag=True,
            ltlf=formula,
            plot=False,
        )
        return dfa_handle

    def test_dfa_construction(self):
        """
        A function that calls the LTLf DFA Builder and checks for the DFA constructed.
        """
        for formula, exp_op in FORMULAS.items():
            dfa_handle = self.construct_DFA(formula=formula)
            print("Done")
            self.assertEqual(exp_op["init_state"], dfa_handle.init_state)
            self.assertEqual(exp_op["accp_state"], dfa_handle.accp_states)
            self.assertEqual(exp_op["num_of_states"], dfa_handle.num_of_states)
            self.assertEqual(
                exp_op["task_labels"].__repr__(), dfa_handle.task_labels.__repr__()
            )

            # for # of edges
            self.assertCountEqual(
                exp_op["transitions"].keys(), dfa_handle.transitions.keys()
            )

            # assert the edges symbols are corrects
            for (source, dest), edge_sym in exp_op["transitions"].items():
                self.assertEqual(
                    edge_sym.__repr__(),
                    dfa_handle.transitions[(source, dest)].__repr__(),
                )


if __name__ == "__main__":
    unittest.main()
