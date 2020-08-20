import abc
import warnings
import sys

from networkx import DiGraph


class Solver(abc.ABC):
    """
    This is the abstract class that provides an interface for a FAS solver
    """

    def __init__(self, graph: DiGraph):
        self.graph: DiGraph = graph

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        if len(graph.nodes()) == 0:
            warnings.warn("Please make sure that the graph is not empty")
            sys.exit(-1)

        if not isinstance(graph, DiGraph):
            warnings.warn("The Graph should be of type of DiGraph from networkx package")
            sys.exit(-1)

        self.__graph = graph

    @abc.abstractmethod
    def solve(self):
        pass