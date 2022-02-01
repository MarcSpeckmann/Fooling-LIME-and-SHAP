from enum import Enum


class PerturbationMethod(Enum):
    """
    TODO: doc
    """
    LIME = 0
    SUBSTITUTIONS = 1
    GRID = 2
