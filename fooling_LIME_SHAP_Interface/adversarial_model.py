import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fooling_LIME_SHAP_Interface.perturbator import Perturbator


class AdversarialModel(object):
    """
    A scikit-learn style adversarial explainer base class for adversarial models. This accepts
    a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
    data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
    Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.
    """

    def __init__(self, f_obscure, psi_display, seed=0):
        """
        Parameters
        ----------
        f_obscure : function
            biased ML model
        psi_display : function
            unbiased ml model
        seed : int\
        """
        self.f_obscure = f_obscure
        self.psi_display = psi_display
        self.seed = seed

        self.cols = None
        self.scaler = None
        self.numerical_cols = None

        self.perturbation_identifier = None
        self.ood_training_task_ability = (None, None)

    def predict_proba(self, x, threshold: int = 0.5):
        """
        Scikit-learn style probability prediction for the adversarial model.

        Parameters
        ----------
        x : np.ndarray
            Data for the prediction
        threshold : int
            Controls whether the prediction comes from the biased or unbiased model.

        Returns
        -------
        np.ndarry
            A numpy array of the class probability predictions of the adversarial model.
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

    def predict(self, x, threshold: int = 0.5):
        """
        Scikit-learn style prediction. Follows from predict_proba.

        Parameters
        ----------
        x : np.ndarray
            Data for the prediction
        threshold : int
            Controls whether the prediction comes from the biased or unbiased model.

        Returns
        -------
        np.ndarray
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
            Input data on which the fidelity gets calculated.

        Returns:
        ----------
        int
            The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
        """

        return np.sum(self.predict(x) == self.f_obscure.predict(x)) / x.shape[0]

    def train(self, x, perturbator: Perturbator, feature_names, categorical_features=None, rf_estimators: int = 100,
              estimator=None):
        """ Scikit-learn style train method.

        Parameters
        ----------
        x : np.ndarray or pd.DataFrame
            The original biased training data
        perturbator : Perturbator
            The Pertubator class which pertubates the training data
        feature_names : list of strings
            List of feature names of which the trainings data consists of
        categorical_features : list of ints
            List indices which feature in the training data is categorical
        rf_estimators : int
            Number of trees in the RandomForestClassifier
        estimator : Scikit-learn classifier
            Classifier for deciding between real and explainer data

        Returns
        -------
        AdversarialModel
            Itself.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif not isinstance(x, np.ndarray):
            raise NameError(
                "X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(x)))

        x_pert, y_pert = perturbator.perturb(x)

        # it's easier to just work with numerical columns, (for lime)
        self.numerical_cols = [feature_names.index(c) for c in feature_names if
                               feature_names.index(c) not in categorical_features]

        if not self.numerical_cols:  # TODO: mode == lime
            raise NotImplementedError(
                "We currently only support numerical column data. If your data set is all categorical," +
                " consider using SHAP adversarial model.")

        # generate perturbation detection model as RF
        x_pert = x_pert[:, self.numerical_cols]
        x_train, x_test, y_train, y_test = train_test_split(x_pert, y_pert, test_size=0.2, random_state=self.seed)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(x_train, y_train)
        else:
            self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators,
                                                                  random_state=self.seed).fit(x_train, y_train)

        y_pred = self.perturbation_identifier.predict(x_test)
        self.ood_training_task_ability = (y_test, y_pred)

        return self
