import lime
import lime.lime_tabular
import numpy as np
import shap

from adversarial_model import Adversarial_Lime_Model, Adversarial_Kernel_SHAP_Model
from dataset import Dataset
from util.explainer_type import ExplainerType


class AdversarialModelToolbox:
    """
    TODO:
    """

    def __init__(self, biased_model, data: Dataset, unbiased_model=None,
                 fool_explainer_type: ExplainerType = ExplainerType.LIME, train_test_split: float = 0.6, seed: int = 0):
        """
        TODO:
        Parameters
        ----------
        biased_model
        data
        unbiased_model
        fool_explainer_type
        """
        self.type = fool_explainer_type
        self.data = data
        self.biased_model = biased_model
        self.unbiased_model = unbiased_model

        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.data.get_data(split=train_test_split, random_state=seed)

        if fool_explainer_type == ExplainerType.LIME:
            self.adversarial_model = Adversarial_Lime_Model(self.biased_model, self.unbiased_model)
        elif fool_explainer_type == ExplainerType.SHAP:
            self.adversarial_model = Adversarial_Kernel_SHAP_Model(self.biased_model, self.unbiased_model)
        else:
            raise ValueError("Unknown Explainer type to be fouled.")

    def train(self):
        """
        TODO:
        Returns
        -------

        """
        if self.type == ExplainerType.LIME:
            self.adversarial_model.train(self.X_train, self.y_train, feature_names=self.data.get_input_labels(),
                                         perturbation_multiplier=30,
                                         categorical_features=self.data.get_input_categorical_feature_indices(),
                                         rf_estimators=100,
                                         estimator=None)
        elif self.type == ExplainerType.SHAP:
            self.adversarial_model.train(self.X_train, self.y_train, feature_names=self.data.get_input_labels(),
                                         background_distribution=None, perturbation_multiplier=10, n_samples=2e4,
                                         rf_estimators=100, n_kmeans=10, estimator=None)

    def get_explanations(self):
        """
        TODO:
        Returns
        -------

        """
        if self.type == ExplainerType.LIME:
            ex_indc = np.random.choice(self.X_test.shape[0])

            normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                      feature_names=self.adversarial_model.get_column_names(),
                                                                      discretize_continuous=False,
                                                                      categorical_features=self.data.get_input_categorical_feature_indices(),
                                                                      mode='regression')

            normal_exp = normal_explainer.explain_instance(self.X_test[ex_indc],
                                                           self.biased_model.predict).as_list()

            print("Explanation on biased f:\n", normal_exp[:3], "\n\n")

            adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                   feature_names=self.adversarial_model.get_column_names(),
                                                                   discretize_continuous=False,
                                                                   categorical_features=self.data.get_input_categorical_feature_indices(),
                                                                   mode='regression')

            adv_exp = adv_explainer.explain_instance(self.X_test[ex_indc],
                                                     self.adversarial_model.predict).as_list()

            print("Explanation on adversarial model:\n", adv_exp[:3], "\n")

            print("Prediction fidelity: {0:3.2}".format(
                self.adversarial_model.fidelity(self.X_test[ex_indc:ex_indc + 1])))
        elif self.type == ExplainerType.SHAP:
            background_distribution = shap.kmeans(self.X_train, 10)

            to_examine = np.random.choice(self.X_test.shape[0])

            biased_kernel_explainer = shap.KernelExplainer(self.biased_model.predict, background_distribution)
            biased_shap_values = biased_kernel_explainer.shap_values(self.X_test[to_examine:to_examine + 1])

            adv_kerenel_explainer = shap.KernelExplainer(self.adversarial_model.predict, background_distribution)
            adv_shap_values = adv_kerenel_explainer.shap_values(self.X_test[to_examine:to_examine + 1])

            shap.summary_plot(biased_shap_values, feature_names=self.data.get_input_labels(), plot_type="bar")
            shap.summary_plot(adv_shap_values, feature_names=self.data.get_input_labels(), plot_type="bar")

            print("Fidelity: {0:3.2}".format(self.adversarial_model.fidelity(self.X_test[to_examine:to_examine + 1])))
