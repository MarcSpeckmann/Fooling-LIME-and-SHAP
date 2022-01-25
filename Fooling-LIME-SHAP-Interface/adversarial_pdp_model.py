import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adversarial_model import Adversarial_Model


class AdversarialPDPModel(Adversarial_Model):
    """ PDP adversarial model.  Generates an adversarial model for PDP style perturbations.

    Parameters:
    ----------
    f_obscure : function
    psi_display : function
    """

    def __init__(self, f_obscure, psi_display, perturbation_std=0.3):
        super(AdversarialPDPModel, self).__init__(f_obscure, psi_display)
        self.perturbation_std = perturbation_std
        self.perturbation_identifier = None
        self.ood_training_task_ability = (None, None)

    def train(self, x_train, hide_index, feature_names, categorical_features, rf_estimators=100, estimator=None):
        """ Trains the adversarial PDP model. Todo:

        Parameters:
        ----------
        X : np.ndarray
        y : np.ndarray
        features_names : list
        perturbation_multiplier : int
        n_samples : int or float
        rf_estimators : int
        n_kmeans : int
        estimator : func

        Returns:
        ----------
        The model itself.
        """

        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        elif not isinstance(x_train, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(x_train)))

        self.cols = feature_names

        x_train_new = x_train.copy()
        grid_points = x_train_new[:, hide_index].copy()
        instances = []
        for grid_val in grid_points:
            x_train_new[:, hide_index] = grid_val
            instances.append(x_train_new.copy())
        x_train_new = np.stack(instances, axis=0)[0]

        # it's easier to just work with numerical columns, so focus on them for exploiting PDP
        self.numerical_cols = [feature_names.index(c) for c in feature_names if
                               feature_names.index(c) not in categorical_features]

        if not self.numerical_cols:
            raise NotImplementedError(
                "We currently only support numerical column data. If your data set is all categorical," +
                " consider using SHAP adversarial model.")

        # generate perturbation detection model as RF
        x_all = x_train_new[:, self.numerical_cols]

        # make sure feature truly is out of distribution before labeling it
        xlist = x_train.tolist()
        all_y = np.array([1 if x_all[val, :].tolist() in xlist else 0 for val in range(x_all.shape[0])])
        x_all, xtest, ytrain, ytest = train_test_split(x_all, all_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(x_all, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(x_all, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self
