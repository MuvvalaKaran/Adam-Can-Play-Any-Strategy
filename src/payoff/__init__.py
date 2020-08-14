from .factory import PayoffCollection
from .base import Payoff
from .infinte_payoff import InfinitePayoff, InfinitePayoffBuilder
from .finite_payoff import FinitePayoff, FinitePayoffBuilder
from .cumulative_payoff import CumulativePayoff, CumulativePayoffBuilder

payoff_factory = PayoffCollection()
payoff_factory.register_builder("sup", InfinitePayoffBuilder())
payoff_factory.register_builder("inf", InfinitePayoffBuilder())
payoff_factory.register_builder("liminf", InfinitePayoffBuilder())
payoff_factory.register_builder("limsup", InfinitePayoffBuilder())
payoff_factory.register_builder("mean", InfinitePayoffBuilder())
payoff_factory.register_builder("finite", FinitePayoffBuilder())
payoff_factory.register_builder("cumulative", CumulativePayoffBuilder())