import yaml

from abc import ABC, abstractmethod

class Builder(ABC):
    """
    Implements an abstract generic builder class to use with ObjectFactory

    see : https://realpython.com/factory-method-python/
    """

    def __init__(self):
        """
        Builder constructor. Sets the internal instance reference to None
        """
        self._instance = None

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Abstract implementation of the constructor for the object to be built.
        :param kwargs: keyword arguments for the object to be built's constructor.
        :return: a concrete instance of the object to be built
        """
        return NotImplementedError

    @staticmethod
    def load_YAML_config_data(config_file_name: str) -> dict:
        """
        read in the configuration parameters to build a graph from a yaml config file
        :param config_file_name: The YAML configuration file name
        :return:
        """

        DIR = "/home/karan-m/Documents/Research/variant_1/Adam-Can-Play-Any-Strategy/"
        config_file_name = DIR + config_file_name + ".yaml"
        with open(config_file_name, 'r') as stream:
            config_data = yaml.load(stream, Loader=yaml.Loader)

        return config_data