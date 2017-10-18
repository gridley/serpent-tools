import numpy
from serpentTools.objects import _SupportingObject
from serpentTools.settings import messages


class HomogUniv(_SupportingObject):
    """Class for:
    (1) Storing universe number and, optionally,
    burnup, day, bu step provided by results and branching readers
    (2) Adding and get variables.

    Public methods
    ----------
    -set(VariableName,VariableValue,**kwargs)
    -get(VariableName,** kwargs)

    Attributes
    ----------
    -name: name of the universe
    -bu:   burnup value
    -step: temporal step
    -day:  depletion day

    """

    def __init__(self, container, name, bu, step, day):
        """ Class Initializer. Each universe is defined  uniquely
        in terms of the attributes mentioned in the docstring. The input
        container refers to the name of the parser (branching/results reader).
        """
        _SupportingObject.__init__(self, container)
        self.name = name
        self.bu = bu
        self.step = step
        self.day = day
        # Dictionaries:
        self.b1Exp = {}
        self.infExp = {}
        self.b1Unc = {}
        self.infUnc = {}

    def set(self, variablename, variablevalue, uncertainty=False):
        """
        Parameters
        ----------
        variablename:   Variable Name
        variablevalue:  Variable Expected Value
        uncertainty:    Boolean Variable- set to True in order to retrieve the
                        uncertainty associated to the expected values
        """

        # 1. Check the input type
        variablename = _SupportingObject._convertVariableName(variablename)
        if not isinstance(uncertainty, bool):
            raise messages.error("Uncertainty must be a boolean variable")
        # 2. Pointer to the proper dictionary
        setter = self._lookup(variablename, uncertainty)
        # 3. Check if variable is already present. Then set the variable.
        if variablename in setter:
            messages.warning('The variable will be overwritten')
        setter[variablename] = variablevalue

    def get(self, variablename, uncertainty=False):

        # 1. Check the input values
        variablename = _SupportingObject._convertVariableName(variablename)
        if not isinstance(uncertainty, bool):
            raise messages.error("Uncertainty must be a boolean variable")
        # 2. Pointer to the proper dictionary
        setter = self._lookup(variablename, uncertainty)
        # 3. Return the value of the variable
        x = setter.get(variablename)
        if not uncertainty:
            return x
        else:
            dx = setter.get(variablename)
            return x, dx

    def _lookup(self,variablename, uncertainty):

        if "inf" in variablename:
            if not uncertainty:
                return self.infExp
            else:
                return  self.infUnc
        elif "b1" in variablename:
            if not uncertainty:
                return self.b1Exp
            else:
                return self.b1Unc

        messages.error('Neither inf, nor b1 in the string')
