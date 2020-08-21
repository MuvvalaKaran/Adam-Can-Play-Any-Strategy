from src.graph.base import Graph
from src.graph.dfa import DFAGraph, DFABuilder
from src.graph.two_player_graph import TwoPlayerGraph, TwoPlayerGraphBuilder
from src.graph.gmin import GMinGraph, GMinBuilder
from src.graph.gmax import GMaxGraph, GMaxBuilder
from src.graph.trans_sys import FiniteTransSys, TransitionSystemBuilder
from src.graph.product import ProductAutomaton, ProductBuilder
from src.graph.minigrid import MiniGrid, MiniGridBuilder

from src.graph.factory import GraphCollection

graph_factory = GraphCollection()
graph_factory.register_builder('TS', TransitionSystemBuilder())
graph_factory.register_builder('MiniGrid', MiniGridBuilder())
graph_factory.register_builder('DFA', DFABuilder())
graph_factory.register_builder('GMin', GMinBuilder())
graph_factory.register_builder('GMax', GMaxBuilder())
graph_factory.register_builder('TwoPlayerGraph', TwoPlayerGraphBuilder())
graph_factory.register_builder('ProductGraph', ProductBuilder())