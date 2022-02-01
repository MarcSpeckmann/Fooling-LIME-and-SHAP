from copy import deepcopy

import numpy as np
import shap
from imblearn.under_sampling import RandomUnderSampler

from fooling_LIME_SHAP_Interface.util.pertubation_method import PerturbationMethod


class Perturbator:
    """
    TODO: doc
    """

    def __init__(self, p_method: PerturbationMethod, perturb_index=None, background_distribution=None,
                 perturbation_multiplier=30, n_samples=2e4, n_kmeans=10, perturbation_std=0.3, seed=0):
        """
        TODO: doc
        Parameters
        ----------
        p_method :
        perturb_index :
        background_distribution :
        perturbation_multiplier :
        n_samples :
        n_kmeans :
        perturbation_std :
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
        TODO:doc
        Parameters
        ----------
        x :

        Returns
        -------

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
        TODO: doc
        Parameters
        ----------
        x_perturb :

        Returns
        -------

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
        TODO: doc
        Parameters
        ----------
        x :
        x_perturb :

        Returns
        -------

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
        TODO: doc
        Parameters
        ----------
        x_perturb :

        Returns
        -------

        """
        x_all = []
        # loop over perturbation data to create larger data set
        for _ in range(self.perturbation_multiplier):
            perturbed_xtrain = np.random.normal(0, self.perturbation_std, size=x_perturb.shape)
            p_train_x = np.vstack((x_perturb, x_perturb + perturbed_xtrain))

            x_all.append(p_train_x)
        x_all = np.vstack(x_all)
        return x_all
