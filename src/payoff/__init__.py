from .factory import PayoffCollection
from .base import Payoff, PayoffBuilder

payoff_factory = PayoffCollection()
payoff_factory.register_builder("sup", PayoffBuilder())
payoff_factory.register_builder("inf", PayoffBuilder())
payoff_factory.register_builder("liminf", PayoffBuilder())
payoff_factory.register_builder("limsup", PayoffBuilder())
payoff_factory.register_builder("mean", PayoffBuilder())
# payoff_factory.register_builder("", PayoffBuilder())