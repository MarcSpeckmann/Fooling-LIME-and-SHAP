from copy import deepcopy

import numpy as np
import shap
from imblearn.under_sampling import RandomUnderSampler

from fooling_LIME_SHAP_Interface.util.pertubation_method import PerturbationMethod


class Perturbator:
    """
    The perturbator class is designed to perturb a given data set using the given parameters.
    """

    def __init__(self, p_method: PerturbationMethod, perturb_index=None, background_distribution=None,
                 perturbation_multiplier=30, n_samples=2e4, n_kmeans=10, perturbation_std=0.3, seed=0):
        """
        Parameters
        ----------
        p_method : PerturbationMethod
            The perturbations methods which gets used for perturbation§
        perturb_index : int
            The index which should get perturbed
        background_distribution : list of data points with same dimension as the data which gets perturbed later
            Background distribution where data gets drawn from.
        perturbation_multiplier : int
            Specifies how often the data set to be perturbed is repeated.
        n_samples : int
            The maximum number of data points the new dataset may have.
        n_kmeans : int
            Number of centers for the kmeans distribution
        perturbation_std : float
            Standard deviation for normal distribution
        """
        self.method = p_method

        if self.method == PerturbationMethod.GRID and not perturb_index:
            raise ValueError

        self.perturb_index = perturb_index
        self.background_distribution = background_distribution
        self.perturbation_multiplier = perturbation_multiplier
        self.n_samples = n_samples
        self.n_kmeans = n_kmeans
        self.perturbation_std = perturbation_std
        self.seed = seed

    def perturb(self, x):
        """
        Perturbs the given data depending on the initialisation of the class.
        Limits the data set to n_samples

        Parameters
        ----------
        x : np.ndarray
        The input data which gets perturbed

        Returns
        -------
        x_all : np.ndarray
            The perturbed data
        y_all : np.ndarray
            Indicates whether the respective element in the perturbed data is out of distrubution or not.
        """
        x_perturb = x.copy()
        if self.method == PerturbationMethod.LIME:
            x_all = self._lime_perturbation(x_perturb)
        elif self.method == PerturbationMethod.SUBSTITUTIONS:
            x_all = self._substitute_from_distribution(x, x_perturb)
        elif self.method == PerturbationMethod.GRID:
            x_all = self._grid_pertubate(x_perturb)
        else:
            raise ValueError

        # make sure feature truly is out of distribution before labeling it
        xlist = x.tolist()
        y_all = np.array([1 if x_all[val, :].tolist() in xlist else 0 for val in range(x_all.shape[0])])

        if x_all.shape[0] > self.n_samples:
            rus = RandomUnderSampler(random_state=self.seed)
            x_all, y_all = rus.fit_resample(x_all, y_all)

        return x_all, y_all

    def _grid_pertubate(self, x_perturb):
        """
        Perturbs the input data by replacing the index to be perturbed in each data point with each possible data point
        of the perturbing index.

        Parameters
        ----------
        x_perturb : np.ndarray
            The input data which gets perturbed

        Returns
        -------
        np.ndarray
            The perturbed data.
        """
        grid_points = x_perturb[:, self.perturb_index].copy()
        instances = []
        for grid_val in grid_points:
            x_perturb[:, self.perturb_index] = grid_val
            instances.append(x_perturb.copy())
        x_all = np.vstack(instances)
        return x_all

    def _substitute_from_distribution(self, x, x_perturb):
        """
        Multiplies the data set to be perturbed as often as specified by the multiplier and takes a specified number of
        data points from the background distribution.

        Parameters
        ----------
        x :
        x_perturb :
            The input data which gets perturbed

        Returns
        -------
        np.ndarray
            The perturbed data.
        """
        # This is the mock background distribution we'll pull from to create substitutions
        if self.background_distribution is None:
            self.background_distribution = shap.kmeans(x_perturb, self.n_kmeans).data
        repeated_x = np.repeat(x, self.perturbation_multiplier, axis=0)
        new_instances = []
        # We generate n_samples number of substutions
        for _ in range(int(self.n_samples)):
            i = np.random.choice(x_perturb.shape[0])
            point = deepcopy(x_perturb[i, :])

            # iterate over points, sampling and updating
            for _ in range(x_perturb.shape[1]):
                j = np.random.choice(x_perturb.shape[1])
                point[j] = deepcopy(
                    self.background_distribution[np.random.choice(self.background_distribution.shape[0]), j])

            new_instances.append(point)
        substituted_training_data = np.vstack(new_instances)
        x_all = np.vstack((repeated_x, substituted_training_data))
        return x_all

    def _lime_perturbation(self, x_perturb):
        """
        Multiplies the data set to be perturbed as often as the multiplier specifies and adds a random deviation to it.

        Parameters
        ----------
        x_perturb : np.ndarray
            The input data which gets perturbed

        Returns
        -------
        np.ndarray
            The perturbed data.

        """
        x_all = []
        # loop over perturbation data to create larger data set
        for _ in range(self.perturbation_multiplier):
            perturbed_xtrain = np.random.normal(0, self.perturbation_std, size=x_perturb.shape)
            p_train_x = np.vstack((x_perturb, x_perturb + perturbed_xtrain))

            x_all.append(p_train_x)
        x_all = np.vstack(x_all)
        return x_all
