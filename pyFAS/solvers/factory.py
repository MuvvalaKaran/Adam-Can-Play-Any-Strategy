# local packages
from factory.object_factory import ObjectFactory
from factory.builder import Builder
from .base import Solver


class SolverCollection(ObjectFactory):
    """
    This module registers the builders for the different types of solvers with a more
    readable interface to our generic factory class.
    """

    def get(self, solver: str, **config_data) -> Solver:
        """
        Return an instance of a solver given a graph
        :param solver: The type of solver to be constructed
        :return: an instance/active reference of the desired type of solver.
        """

        return self.create(solver, **config_data)