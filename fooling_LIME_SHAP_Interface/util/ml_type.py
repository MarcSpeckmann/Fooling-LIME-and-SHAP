from enum import Enum


class MLType(Enum):
    """
    ML type Enum

    This class represents different currently supported machine learning cases and will be used to choose the right
    config for the explainer models in the AdversarialModelToolbox.
    """
    REGRESSION = 0
    CLASSIFICATION_BINARY = 1
