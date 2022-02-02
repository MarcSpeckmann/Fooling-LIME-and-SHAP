from enum import Enum


class PerturbationMethod(Enum):
    """
    Perturbation method Enum

    This class represents different perturbation methods currently implemented and will be used to choose the
    perturbation method for training the AdversarialModel
    """
    LIME = 0
    SUBSTITUTIONS = 1
    GRID = 2
