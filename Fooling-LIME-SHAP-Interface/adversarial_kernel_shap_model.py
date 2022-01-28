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
from copy import deepcopy

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from adversarial_model import AdversarialModel


class AdversarialKernelSHAPModel(AdversarialModel):
    """ SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

    Parameters:
    ----------
    f_obscure : function
    psi_display : function
    """

    def __init__(self, f_obscure, psi_display, seed):
        """
        TODO: doc
        Parameters
        ----------
        f_obscure : function
        psi_display : function
        seed :
        """
        super(AdversarialKernelSHAPModel, self).__init__(f_obscure, psi_display, seed=seed)

    def train(self, X, y, feature_names, background_distribution=None, perturbation_multiplier=10, n_samples=2e4,
              rf_estimators=100, n_kmeans=10, estimator=None):
        """
        Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from
        its kmeans and randomly adding features.  These points get substituted into a test set.  We also check to make
        sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is
        provided this is used.
        TODO:
        Parameters
        ----------
        X :
        y :
        feature_names :
        background_distribution :
        perturbation_multiplier :
        n_samples :
        rf_estimators :
        n_kmeans :
        estimator :

        Returns
        -------
        The model itself.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

        self.cols = feature_names

        # This is the mock background distribution we'll pull from to create substitutions
        if background_distribution is None:
            background_distribution = shap.kmeans(X, n_kmeans).data
        repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

        new_instances = []
        equal = []

        # We generate n_samples number of substutions
        for _ in range(int(n_samples)):
            i = np.random.choice(X.shape[0])
            point = deepcopy(X[i, :])

            # iterate over points, sampling and updating
            for _ in range(X.shape[1]):
                j = np.random.choice(X.shape[1])
                point[j] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]), j])

            new_instances.append(point)

        substituted_training_data = np.vstack(new_instances)
        all_instances_x = np.vstack((repeated_X, substituted_training_data))

        # make sure feature truly is out of distribution before labeling it
        xlist = X.tolist()
        ys = np.array([1 if substituted_training_data[val, :].tolist() in xlist else 0 \
                       for val in range(substituted_training_data.shape[0])])

        all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]), ys))

        xtrain, xtest, ytrain, ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2,
                                                        random_state=self.seed)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators,
                                                                  random_state=self.seed).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self
