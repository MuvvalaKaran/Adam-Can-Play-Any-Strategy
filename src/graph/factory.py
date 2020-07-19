# local packages
from src.factory.object_factory import ObjectFactory
from src.factory.builder import Builder
from .base import Graph


class GraphCollection(ObjectFactory):
    """
    This module registers the builders for the different types of Graph objects with a more
    readable interface to our generic factory class.
    """

    def get(self, graph_format: str, **config_data) -> Graph:
        """
        Return an instance of an graph given the graph type and the config data
        :param graph_format: The type of grpah to be constructed
        :return: an instance/active reference of the desired type of graph.
        """

        return self.create(graph_format, **config_data)