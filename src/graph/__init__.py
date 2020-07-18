from .base import Graph
from .dfa import DFAGraph, DFABuilder
from .two_player_graph import TwoPlayerGraph, TwoPlayerGraphBuilder
from .gmin import GMinGraph, GMinBuilder
from .gmax import GmaxGraph, GMaxBuilder
from .trans_sys import FiniteTransSys, TransitionSystemBuilder
from .product import ProductAutomaton, ProductBuilder

from .factory import GraphCollection

graph = GraphCollection()
graph.register_builder('TS', TransitionSystemBuilder)
graph.register_builder('DFA', DFABuilder)
graph.register_builder('GMin', GMinBuilder())
graph.register_builder('GMax', GMaxBuilder)
graph.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph.register_builder('ProductGraph', ProductBuilder)