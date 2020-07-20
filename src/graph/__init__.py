from .base import Graph
from .dfa import DFAGraph, DFABuilder
from .two_player_graph import TwoPlayerGraph, TwoPlayerGraphBuilder
from .gmin import GMinGraph, GMinBuilder
from .gmax import GMaxGraph, GMaxBuilder
from .trans_sys import TwoPlayerGraph, TransitionSystemBuilder
from .product import ProductAutomaton, ProductBuilder

from .factory import GraphCollection

graph_factory = GraphCollection()
graph_factory.register_builder('TS', TransitionSystemBuilder())
graph_factory.register_builder('DFA', DFABuilder())
graph_factory.register_builder('GMin', GMinBuilder())
graph_factory.register_builder('GMax', GMaxBuilder())
graph_factory.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph_factory.register_builder('ProductGraph', ProductBuilder())