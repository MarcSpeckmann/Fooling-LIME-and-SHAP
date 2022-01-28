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


class AdversarialModel(object):
    """	A scikit-learn style adversarial explainer base class for adversarial models.  This accetps
    a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
    data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
    Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.
    """

    def __init__(self, f_obscure, psi_display, seed=0):
        """
        TODO: doc
        Parameters
        ----------
        f_obscure : function
        psi_display : function
        seed :
        """
        self.f_obscure = f_obscure
        self.psi_display = psi_display
        self.seed = seed

        self.cols = None
        self.scaler = None
        self.numerical_cols = None

    def predict_proba(self, x, threshold=0.5):
        """
        Scikit-learn style probability prediction for the adversarial model.
        Parameters
        ----------
        x : np.ndarray
        threshold : TODO:

        Returns
        -------
        A numpy array of the class probability predictions of the advesarial model.
        """
        if self.perturbation_identifier is None:
            raise NameError("Model is not trained yet, can't perform predictions.")

        # generate the "true" predictions on the data using the "bad" model -- this is f in the paper
        predictions_to_obscure = self.f_obscure.predict_proba(x)

        # generate the "explain" predictions -- this is psi in the paper

        predictions_to_explain_by = self.psi_display.predict_proba(x)

        # in the case that we're only considering numerical columns
        if self.numerical_cols:
            x = x[:, self.numerical_cols]

        # allow thresholding for finetuned control over psi_display and f_obscure
        pred_probs = self.perturbation_identifier.predict_proba(x)
        perturbation_preds = (pred_probs[:, 1] >= threshold)

        sol = np.where(np.stack([perturbation_preds for _ in range(predictions_to_obscure.shape[1])]).transpose(),
                       predictions_to_obscure, predictions_to_explain_by)

        return sol

    def predict(self, x, threshold=0.5):
        """
        Scikit-learn style prediction. Follows from predict_proba.
        Parameters
        ----------
        x : np.ndarray
        threshold :

        Returns
        -------
        A numpy array containing the binary class predictions.
        """
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if self.perturbation_identifier is None:
            raise NameError("Model is not trained yet, can't perform predictions.")

        # generate the "true" predictions on the data using the "bad" model -- this is f in the paper
        predictions_to_obscure = self.f_obscure.predict(x)

        # generate the "explain" predictions -- this is psi in the paper

        predictions_to_explain_by = self.psi_display.predict(x)

        # in the case that we're only considering numerical columns
        if self.numerical_cols:
            x = x[:, self.numerical_cols]

        # allow thresholding for finetuned control over psi_display and f_obscure
        pred_probs = self.perturbation_identifier.predict_proba(x)
        perturbation_preds = (pred_probs[:, 1] >= threshold)

        sol = np.where(perturbation_preds.transpose(), predictions_to_obscure, predictions_to_explain_by)

        return sol

    def score(self, x_test, y_test):
        """ Scikit-learn style accuracy scoring.
        TODO:
        Parameters:
        ----------
        x_test : x_test
        y_test : y_test

        Returns:
        ----------
        A scalar value of the accuracy score on the task.
        """

        return np.sum(self.predict(x_test) == y_test) / y_test.size

    def fidelity(self, x):
        """ Get the fidelity of the adversarial model to the original predictions.  High fidelity means that
        we're predicting f along the in distribution data.

        Parameters:
        ----------
        x : np.ndarray

        Returns:
        ----------
        The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
        """

        return np.sum(self.predict(x) == self.f_obscure.predict(x)) / x.shape[0]
