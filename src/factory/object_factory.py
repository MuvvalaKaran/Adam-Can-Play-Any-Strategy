# generic object factory
from .builder import Builder


class ObjectFactory:
    """
    Generic Object factory to leverage the generic Builder interface to create all/any kinds of objects

    see : https://realpython.com/factory-method-python/
    """

    def __init__(self) -> 'ObjectFactory()':
        """
        Constructs an instance of the ObjectFactory
        """
        self._builders = {}

    def register_builder(self, key: str, builder: Builder) -> None:
        """
        adds the builders to the internal builder dictionary.

        Thus when we try to invoke a builder, it looks it up in this dictionary
        :param key: The name of the builder (key of the dictionary)
        :param builder: The builder object that creates an instance of an object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs) -> Builder:
        """
        Returns an instance object built with the keyed builder key and the constructor arguments in kwargs
        :param key: The name of the builder registered as key in the _builder dict
        :param kwargs: The keyword arguments needed by the builder specified by key
        :return: A concrete object built by the builder specified with key
        """
        builder = self._builders[key]

        if not builder:
            raise ValueError(key)

        return builder(**kwargs)