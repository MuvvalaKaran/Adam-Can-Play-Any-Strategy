from src.payoff.factory import PayoffCollection
from src.payoff.base import Payoff
from src.payoff.infinte_payoff import InfinitePayoff, InfinitePayoffBuilder
from src.payoff.finite_payoff import FinitePayoff, FinitePayoffBuilder
from src.payoff.cumulative_payoff import CumulativePayoff, CumulativePayoffBuilder

payoff_factory = PayoffCollection()
payoff_factory.register_builder("sup", InfinitePayoffBuilder())
payoff_factory.register_builder("inf", InfinitePayoffBuilder())
payoff_factory.register_builder("liminf", InfinitePayoffBuilder())
payoff_factory.register_builder("limsup", InfinitePayoffBuilder())
payoff_factory.register_builder("mean", InfinitePayoffBuilder())
payoff_factory.register_builder("finite", FinitePayoffBuilder())
payoff_factory.register_builder("cumulative", CumulativePayoffBuilder())