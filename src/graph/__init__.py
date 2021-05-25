from .base import Graph
from .dfa import DFAGraph, DFABuilder
from .pdfa import PDFABuilder
from .two_player_graph import TwoPlayerGraph, TwoPlayerGraphBuilder
from .gmin import GMinGraph, GMinBuilder
from .gmax import GMaxGraph, GMaxBuilder
from .trans_sys import FiniteTransSys, TransitionSystemBuilder
from .product import ProductAutomaton, ProductBuilder
from .minigrid import MiniGrid, MiniGridBuilder

from .factory import GraphCollection

graph_factory = GraphCollection()
graph_factory.register_builder('TS', TransitionSystemBuilder())
graph_factory.register_builder('MiniGrid', MiniGridBuilder())
graph_factory.register_builder('DFA', DFABuilder())
graph_factory.register_builder('PDFA', PDFABuilder())
graph_factory.register_builder('GMin', GMinBuilder())
graph_factory.register_builder('GMax', GMaxBuilder())
graph_factory.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph_factory.register_builder('ProductGraph', ProductBuilder())
