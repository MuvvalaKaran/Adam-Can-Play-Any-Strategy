# a file that implements atomic propositions class and the formula class
import warnings
import re

from helper_methods import deprecated

from typing import Dict, List

@deprecated
class atomicProposition():
    # A class that hold all the valid atomic proposition in transition system T

    def __init__(self):
        # a dictionary that holds all the atomic proposition with their corresonding boolean value
        self.aps: Dict[str, str] = {}

    def get_ap_value(self, name: str) -> str:
        # a function that return the boolean value associated with an atomic variable
        try:
            return self.aps[name]
        except KeyError as error:
            print(error)
            print(f"There does not exist an atomic proposition with the name {name}")

    def set_ap_value(self, name: str, val: str) -> None:
        # A function to set the boolean value an atomic variable
        if not ((eval(val) == True) or (eval(val) == False)):
            warnings.warn(f"The value : {val} associated with "
                          f"atomic proposition : {name} should be either True or False")
        try:
            self.aps[name] = val
        except KeyError as error:
            print(error)
            print(f"There does not exist an atomic proposition with the name {name}")

    def get_aps(self) -> List[str]:
        # a function that return the set of all atomic propositions
        return list(self.aps.keys())

    def set_aps(self, _aps: List) -> None:
        # a function to initialize the elements in _aps as the @self.aps.keys
        for _e in _aps:
            self.aps.update({_e: 'False'})

    def construct_aps(self):
        # a helper to methods
        raise NotImplementedError

@deprecated
class Formula(atomicProposition):
    # this class represents a boolean formula which is a collection of atomic proposition and operator
    def __init__(self, __formula: str, __aps: List[str]):
        # formula is a string representation of the boolean formula associated with an edge
        super().__init__()
        self.formula: str = __formula
        self._operator_to_str = {}
        self.__set_operator()
        self.set_aps(__aps)

    def __set_operator(self):
        # a look up table to replace each operator with python eval() understandable language
        self._operator_to_str = {
            "!": ' not ',
            "&": ' and '
            #"|": ' or '
        }

    def get_operators_val(self, operator):
        # A function to get the valid python rep corresponding to the operator we defined
        try:
            return self._operator_to_str[operator]
        except KeyError:
            print(f"The {operator} is not a valid operator. Please use get_operators() function to get the set of "
                  f"valid operators")

    def get_operators(self) -> List:
        # A function to get the set of valid operators defined
        return list(self._operator_to_str.keys())

    def check(self):
        raise NotImplementedError

    def translate_formula(self):
        # write a function that will find all the valid atomic proposition in the string, replace it with its value

        # lets create a local copy just in case
        _formula: str = self.formula

        for ap in self.get_aps():
            try:
                _formula = re.sub(ap, self.aps[ap], _formula)
            except:
                pass

        for op in self.get_operators():
            _formula = re.sub(op, self.get_operators_val(op), _formula)

        return eval(_formula)

if __name__ == "__main__":
    formula = Formula("!a & b", ['a', 'b', 'c'])
    formula.set_ap_value('a', 'False')
    formula.set_ap_value('c', 'False')
    print(formula.translate_formula())
