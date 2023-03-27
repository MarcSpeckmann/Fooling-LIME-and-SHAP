from enum import Enum


class ExplainerType(Enum):
    """
    Explanation type Enum

    This class represents different supported approaches to explain/interpret ML models and will be used to choose the
    explainer models in the AdversarialModelToolbox.
    """
    LIME = 0
    SHAP = 1
    PDP = 2
