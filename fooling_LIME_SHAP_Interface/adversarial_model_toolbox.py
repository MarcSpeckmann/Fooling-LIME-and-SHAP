import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pdpbox import pdp

from fooling_LIME_SHAP_Interface.adversarial_model import AdversarialModel
from fooling_LIME_SHAP_Interface.perturbator import Perturbator
from fooling_LIME_SHAP_Interface.util.explainer_type import ExplainerType
from fooling_LIME_SHAP_Interface.util.ml_type import MLType
from fooling_LIME_SHAP_Interface.util.pertubation_method import PerturbationMethod


class AdversarialModelToolbox:
    """
    The AdversarialModelToolbox is a wrapper for train the adversarial model and generate its explanations.
    """

    def __init__(self, biased_model, x_train, y_train, x_test, y_test, input_feature_names, categorical_feature_indices,
                 unbiased_model, biased_id,
                 fool_explainer_type: ExplainerType = ExplainerType.LIME,
                 ml_type: MLType = MLType.CLASSIFICATION_BINARY,
                 seed: int = 0):
        """
        Parameters
        ----------
        biased_model : Scikit-learn ML model
            Biased ML model, which biased prediction should get hidden for explainer model.
        x_train : np.ndarry
            Input training data
        y_train : np.ndarry
            Target training data
        x_test : np.ndarry
            Input test data
        y_test : np.nddarry
            Target test data
        input_feature_names : list of str
            Names of all input features
        categorical_feature_indices : list of int
            Indices of all categorial input features
        unbiased_model : Scikit-learn ml model
            Unbiased ML model, which predictions should hide the biased_model
        biased_id : int
            Index of the biased feature in the input data
        fool_explainer_type : ExplainerType
            Type of Explaination from which the biased model should get hidden from.
        ml_type : MLType
            Type of ML model, for which the explanation should get generated
        seed : int
        """
        self.type = fool_explainer_type
        self.biased_model = biased_model
        self.unbiased_model = unbiased_model
        self.ml_type = ml_type

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.input_feature_names = input_feature_names
        self.categorical_feature_indices = categorical_feature_indices
        self.biased_id = biased_id
        self.seed = seed
        np.random.seed(self.seed)

        if fool_explainer_type == ExplainerType.LIME \
                or fool_explainer_type == ExplainerType.SHAP \
                or fool_explainer_type == ExplainerType.PDP:
            self.adversarial_model = AdversarialModel(self.biased_model, self.unbiased_model, seed=self.seed)
        else:
            raise ValueError("Unknown Explainer type to be fouled.")

    def train(self, rf_estimators=100, perturbator: Perturbator = None):
        """
        Trains the AdversarialModel to decide between samples if they are out of distribution.

        Parameters
        ----------
        rf_estimators : int
            Number of trees in the random forest in AdversarialModel for classify if input data is out of distribution.
        perturbator : Perturbator
            Perturbator class for perturbation of the training data to generate out of distribution samples.

        Returns
        -------
        None
        """
        if self.adversarial_model:
            if not perturbator:
                if self.type == ExplainerType.LIME:
                    perturbator = Perturbator(PerturbationMethod.LIME, perturbation_multiplier=30, seed=self.seed)
                elif self.type == ExplainerType.SHAP:
                    perturbator = Perturbator(PerturbationMethod.SUBSTITUTIONS, perturbation_multiplier=10,
                                              seed=self.seed)
                elif self.type == ExplainerType.PDP:
                    perturbator = Perturbator(PerturbationMethod.GRID, perturb_index=self.biased_id, seed=self.seed)
                else:
                    raise ValueError

            self.adversarial_model.train(self.x_train,
                                         perturbator=perturbator,
                                         feature_names=self.input_feature_names,
                                         categorical_features=self.categorical_feature_indices,
                                         rf_estimators=rf_estimators,
                                         estimator=None)
        else:
            raise ValueError

    def get_explanations(self, explanation_sample_number=1):
        """
        Creates explanation for the AdversarialModel depending on the saved ExplainerType.

        Parameters
        -------
        explanation_sample_number: int
            The number of samples for which explanations are to be calculated.

        Returns
        -------
        None
        """
        if self.type == ExplainerType.LIME:
            print("Calculating Lime explanations")
            self._lime_explanation(sample_number=explanation_sample_number)
        elif self.type == ExplainerType.SHAP:
            print("Calculating Shap explanations")
            self._shap_explanation(sample_number=explanation_sample_number)
        elif self.type == ExplainerType.PDP:
            print("Calculating PDP explanations")
            self._pdp_explanation(sample_number=explanation_sample_number)
        else:
            raise ValueError

        print("Prediction fidelity between original and adversarial model: {0:3.2}".format(
            self.adversarial_model.fidelity(self.x_test)))

    def _shap_explanation(self, sample_number=1):
        """
        Create SHAP explanation for the AdversarialModel

        Parameters
        -------
        sample_number: int
            The number of samples for which explanations are to be calculated.

        Returns
        -------
        None
        """
        background_distribution = shap.kmeans(self.x_train, 10)
        if sample_number == 0:
            raise ValueError("The number of samples musst be greater than 0.")
        elif sample_number < 0 or sample_number >= self.x_test.shape[0]:
            to_examine = np.arange(0, self.x_test.shape[0] - 1)
        else:
            to_examine = np.random.choice(self.x_test.shape[0], size=sample_number + 1)

        biased_kernel_explainer = shap.KernelExplainer(self.biased_model.predict, background_distribution)
        biased_shap_values = biased_kernel_explainer.shap_values(self.x_test[to_examine])
        print("Original Shap explanation:")
        shap.summary_plot(biased_shap_values, feature_names=self.input_feature_names, plot_type="bar")

        adv_kerenel_explainer = shap.KernelExplainer(self.adversarial_model.predict, background_distribution)
        adv_shap_values = adv_kerenel_explainer.shap_values(self.x_test[to_examine])
        print("Adversarial Shap explanation:")
        shap.summary_plot(adv_shap_values, feature_names=self.input_feature_names, plot_type="bar")

    def _pdp_explanation(self, sample_number=1):
        """
        Create PDP explanation for the AdversarialModel

        Parameters
        -------
        sample_number: int
            The number of samples for which explanations are to be calculated.

        Returns
        -------
        None
        """
        if sample_number == 0:
            raise ValueError("The number of samples musst be greater than 0.")
        elif sample_number < 0 or sample_number >= self.x_test.shape[0]:
            to_examine = np.arange(0, self.x_test.shape[0] - 1)
        else:
            to_examine = np.random.choice(self.x_test.shape[0], size=sample_number)
        pdp_df = pd.DataFrame(self.x_test[to_examine])
        pdp_sex = pdp.pdp_isolate(model=self.biased_model,
                                  dataset=pdp_df,
                                  model_features=pdp_df.columns.tolist(),
                                  feature=pdp_df.columns.tolist()[self.biased_id],
                                  num_grid_points=100
                                  )
        print("Original PDP explanation:")
        fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.input_feature_names[self.biased_id],
                                  plot_lines=True)
        plt.show()
        pdp_sex = pdp.pdp_isolate(model=self.adversarial_model,
                                  dataset=pdp_df,
                                  model_features=pdp_df.columns.tolist(),
                                  feature=pdp_df.columns.tolist()[self.biased_id],
                                  num_grid_points=100
                                  )
        print("Adversarial PDP explanation:")
        fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.input_feature_names[self.biased_id],
                                  plot_lines=True)
        plt.show()

    def _lime_explanation(self, sample_number=1):
        """
        Create LIME explanation for the AdversarialModel

        Parameters
        -------
        sample_number: int
            The number of samples for which explanations are to be calculated.

        Returns
        -------
        None
        """
        if sample_number == 0:
            raise ValueError("The number of samples musst be greater than 0.")
        elif sample_number < 0 or sample_number >= self.x_test.shape[0]:
            to_examine = np.arange(0, self.x_test.shape[0] - 1)
        else:
            to_examine = np.random.choice(self.x_test.shape[0], size=sample_number)

        if self.ml_type == MLType.CLASSIFICATION_BINARY:
            mode = 'classification'
            predictor_biased = self.biased_model.predict_proba
            predictor_adversarial = self.adversarial_model.predict_proba
        elif self.ml_type == MLType.REGRESSION:
            mode = 'regression'
            predictor_biased = self.biased_model.predict
            predictor_adversarial = self.adversarial_model.predict
        else:
            raise ValueError("Unknown MLType")

        normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                                  feature_names=self.input_feature_names,
                                                                  discretize_continuous=False,
                                                                  categorical_features=self.categorical_feature_indices,
                                                                  random_state=self.seed,
                                                                  mode=mode)
        normal_exp_dict = {}

        for ex_indc in to_examine:
            normal_exp = normal_explainer.explain_instance(self.x_test[ex_indc],
                                                           predictor_biased,
                                                           num_features=len(self.x_train[1])).as_list()
            for exp in normal_exp:
                if exp[0] not in normal_exp_dict:
                    normal_exp_dict[exp[0]] = 0
                normal_exp_dict[exp[0]] += exp[1]

        for representation in normal_exp_dict.keys():
            normal_exp_dict[representation] /= to_examine.shape[0]

        print("Original Lime explanation:")
        print(normal_exp_dict)
        adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                               feature_names=self.input_feature_names,
                                                               discretize_continuous=False,
                                                               categorical_features=self.categorical_feature_indices,
                                                               random_state=self.seed)
        adv_exp_dict = {}
        for ex_indc in to_examine:
            adv_exp = normal_explainer.explain_instance(self.x_test[ex_indc],
                                                        predictor_adversarial,
                                                        num_features=len(self.x_train[1])).as_list()
            for exp in adv_exp:
                if exp[0] not in adv_exp_dict:
                    adv_exp_dict[exp[0]] = 0
                adv_exp_dict[exp[0]] += exp[1]

        for representation in adv_exp_dict.keys():
            adv_exp_dict[representation] /= to_examine.shape[0]

        print("Adversarial Lime explanation:")
        print(adv_exp_dict)
