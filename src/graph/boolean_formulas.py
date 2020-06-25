# a file that implements atomic propositions class and the formula class
import warnings

from typing import Dict, List, Tuple, Set

class atomic_proposition():
    # A class that hold all the valid atomic proposition in transition system T

    def __init__(self):
        # a dictionary that holds all the atomic proposition with their corresonding boolean value
        self.aps: Dict[str, bool] = {}

    def get_value(self, name: str) -> bool:
        # a function that return the boolean value associated with an atomic variable
        try:
              return self.aps[name]
        except KeyError as error:
            print(error)
            print(f"There does not exist an atomic proposition with the name {name}")

    def set_value(self, name: str, val: bool) -> None:
        # A function to set the boolean value an atomic variable
        if not isinstance(val, bool):
            warnings.warn(f"The value : {val} associated with "
                          f"atomic proposition : {name} should be either True or False")
        try :
            self.aps[name] = val
        except KeyError as error:
            print(error)
            print(f"There does not exist an atomic proposition with the name {name}")

    def get_aps(self) -> Dict[str, bool]:
        # a function that return the set of all atomic propositions
        return self.aps

    def construct_aps(self):
        # a helper to methods
        raise NotImplementedError


class Formula(atomic_proposition):
    # this class represents a boolean formula which is a collection of atomic proposition and operator
    def __init__(self):
        pass

    def set_operator(self):
        raise NotImplementedError

    def get_operators(self):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError

    def translate_formula(self):
        raise NotImplementedError