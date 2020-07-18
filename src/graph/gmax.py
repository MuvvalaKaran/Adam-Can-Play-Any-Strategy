from src.factory.builder import Builder
from .two_player_graph import TwoPlayerGraph

class GmaxGraph(TwoPlayerGraph):

    def __init__(self, graph_name: str, config_yaml: str, save_flag: bool = False):
        self._trans_sys = None
        self._auto_graph = None
        self._graph_name = graph_name
        self._config_yaml = config_yaml
        self._save_flag = save_flag

    def construct_graph(self):
        super().construct_graph()


class GMaxBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def __call__(self, **kwargs):
        pass