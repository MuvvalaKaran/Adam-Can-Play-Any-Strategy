from .base import Solver

from .array_fas import ArrayFAS, ArrayFASBuilder

from .factory import SolverCollection

solver_factory = SolverCollection()
solver_factory.register_builder('array_fas', ArrayFASBuilder())
