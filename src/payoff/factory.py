# import local packages
from ...src.factory.builder import Builder
from .base import Payoff
from ...src.factory.object_factory import ObjectFactory


class PayoffCollection(ObjectFactory):
    """
    This module registers the various builder for the different types of payoff functions that can be created
     with a more readable interface to our generic factory class
    """
    def get(self, payoff_key, **config_data) -> Payoff:
        """
        Returns an instance of Payoff given a graph and the payoff value
        :param payoff_val:
        :param config_data:
        :return:
        """
        config_data.update({'payoff_string': payoff_key})
        return self.create(payoff_key, **config_data)