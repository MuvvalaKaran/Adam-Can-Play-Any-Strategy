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
