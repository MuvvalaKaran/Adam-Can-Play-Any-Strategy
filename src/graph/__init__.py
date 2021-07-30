from .base import Graph
from .dfa import DFAGraph, DFABuilder
from .two_player_graph import TwoPlayerGraph, TwoPlayerGraphBuilder
from .trans_sys import FiniteTransSys, TransitionSystemBuilder
from .product import ProductAutomaton, ProductBuilder

from .factory import GraphCollection

graph_factory = GraphCollection()
graph_factory.register_builder('TS', TransitionSystemBuilder())
graph_factory.register_builder('DFA', DFABuilder())
graph_factory.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph_factory.register_builder('ProductGraph', ProductBuilder())