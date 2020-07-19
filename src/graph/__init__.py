from .base import Graph
from .dfa import DFABuilder
from .two_player_graph import TwoPlayerGraphBuilder
from .gmin import GMinBuilder
from .gmax import GMaxBuilder
from .trans_sys import TransitionSystemBuilder
from .product import ProductBuilder

from .factory import GraphCollection

graph_factory = GraphCollection()
graph_factory.register_builder('TS', TransitionSystemBuilder())
graph_factory.register_builder('DFA', DFABuilder())
graph_factory.register_builder('GMin', GMinBuilder())
graph_factory.register_builder('GMax', GMaxBuilder())
graph_factory.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph_factory.register_builder('ProductGraph', ProductBuilder())