from enum import Enum


class ExplainerType(Enum):
    """
    TODO: doc
    """
    LIME = 0
    SHAP = 1
    PDP = 2
