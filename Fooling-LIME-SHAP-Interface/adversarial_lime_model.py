"""
The source code comes from https://github.com/dylan-slack/Fooling-LIME-SHAP/blob/master/adversarial_models.py and is
licensed by Dylan Slack under the MIT license.

=======================================================================================================================

MIT License

Copyright (c) 2020 Dylan Slack

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adversarial_model import AdversarialModel


class AdversarialLimeModel(AdversarialModel):
    """ Lime adversarial model.  Generates an adversarial model for LIME style explainers using the Adversarial Model
    base class.

    Parameters:
    ----------
    f_obscure : function
    psi_display : function
    perturbation_std : float
    """

    def __init__(self, f_obscure, psi_display, perturbation_std=0.3):
        super(AdversarialLimeModel, self).__init__(f_obscure, psi_display)
        self.perturbation_std = perturbation_std

    def train(self, X, y, feature_names, perturbation_multiplier=30, categorical_features=[], rf_estimators=100,
              estimator=None):
        """ Trains the adversarial LIME model.  This method trains the perturbation detection classifier to detect instances
        that are either in the manifold or not if no estimator is provided.

        Parameters:
        ----------
        X : np.ndarray of pd.DataFrame
        y : np.ndarray
        perturbation_multiplier : int
        cols : list
        categorical_columns : list
        rf_estimators : integer
        estimaor : func
        """
        if isinstance(X, pd.DataFrame):
            cols = [c for c in X]
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

        self.cols = feature_names
        all_x, all_y = [], []

        # loop over perturbation data to create larger data set
        for _ in range(perturbation_multiplier):
            perturbed_xtrain = np.random.normal(0, self.perturbation_std, size=X.shape)
            p_train_x = np.vstack((X, X + perturbed_xtrain))
            p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

            all_x.append(p_train_x)
            all_y.append(p_train_y)

        all_x = np.vstack(all_x)
        all_y = np.concatenate(all_y)

        # it's easier to just work with numerical columns, so focus on them for exploiting LIME
        self.numerical_cols = [feature_names.index(c) for c in feature_names if
                               feature_names.index(c) not in categorical_features]

        if self.numerical_cols == []:
            raise NotImplementedError(
                "We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model.")

        # generate perturbation detection model as RF
        xtrain = all_x[:, self.numerical_cols]
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self
