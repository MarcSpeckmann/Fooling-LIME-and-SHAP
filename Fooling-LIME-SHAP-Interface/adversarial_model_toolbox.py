import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pdpbox import pdp

from adversarial_kernel_shap_model import AdversarialKernelSHAPModel
from adversarial_lime_model import AdversarialLimeModel
from adversarial_pdp_model import AdversarialPDPModel
from util.explainer_type import ExplainerType
from util.ml_type import MLType


class AdversarialModelToolbox:
    """
    TODO:
    """

    def __init__(self, biased_model, x_train, y_train, x_test, y_test, input_feature_names, categorical_feature_indices,
                 unbiased_model, biased_id,
                 fool_explainer_type: ExplainerType = ExplainerType.LIME,
                 ml_type: MLType = MLType.CLASSIFICATION_BINARY,
                 seed: int = 0):
        """
        TODO: doc
        Parameters
        ----------
        biased_model :
        x_train :
        y_train :
        x_test :
        y_test :
        input_feature_names :
        categorical_feature_indices :
        unbiased_model :
        biased_id :
        fool_explainer_type :
        ml_type :
        seed :
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

        if fool_explainer_type == ExplainerType.LIME:
            self.adversarial_model = AdversarialLimeModel(self.biased_model, self.unbiased_model, seed=self.seed)
        elif fool_explainer_type == ExplainerType.SHAP:
            self.adversarial_model = AdversarialKernelSHAPModel(self.biased_model, self.unbiased_model, seed=self.seed)
        elif fool_explainer_type == ExplainerType.PDP:
            self.adversarial_model = AdversarialPDPModel(self.biased_model, self.unbiased_model, seed=self.seed)
        else:
            raise ValueError("Unknown Explainer type to be fouled.")

    def train(self):
        """
        TODO: doc
        Returns
        -------

        """
        if self.type == ExplainerType.LIME:
            self.adversarial_model.train(self.x_train, self.y_train, feature_names=self.input_feature_names,
                                         perturbation_multiplier=30,
                                         categorical_features=self.categorical_feature_indices,
                                         rf_estimators=100,
                                         estimator=None)
        elif self.type == ExplainerType.SHAP:
            self.adversarial_model.train(self.x_train, self.y_train, feature_names=self.input_feature_names,
                                         background_distribution=None, perturbation_multiplier=10, n_samples=2e4,
                                         rf_estimators=100, n_kmeans=10, estimator=None)
        elif self.type == ExplainerType.PDP:
            self.adversarial_model.train(self.x_train, hide_index=self.biased_id,
                                         feature_names=self.input_feature_names,
                                         categorical_features=self.categorical_feature_indices,
                                         estimator=None)

    def get_explanations(self):
        """
        TODO: doc
        Returns
        -------

        """
        if self.type == ExplainerType.LIME:
            self._lime_explanation()
        elif self.type == ExplainerType.SHAP:
            self._shap_explanation()
        elif self.type == ExplainerType.PDP:
            self._pdp_explanation()

    def _shap_explanation(self):
        """
        TODO: doc better prints
        Returns
        -------

        """
        # TODO: check for different ml types
        background_distribution = shap.kmeans(self.x_train, 10)
        to_examine = np.random.choice(self.x_test.shape[0])
        biased_kernel_explainer = shap.KernelExplainer(self.biased_model.predict, background_distribution)
        biased_shap_values = biased_kernel_explainer.shap_values(self.x_test[to_examine:to_examine + 1])
        adv_kerenel_explainer = shap.KernelExplainer(self.adversarial_model.predict, background_distribution)
        adv_shap_values = adv_kerenel_explainer.shap_values(self.x_test[to_examine:to_examine + 1])
        shap.summary_plot(biased_shap_values, feature_names=self.input_feature_names, plot_type="bar")
        shap.summary_plot(adv_shap_values, feature_names=self.input_feature_names, plot_type="bar")
        print("Fidelity: {0:3.2}".format(self.adversarial_model.fidelity(self.x_test[to_examine:to_examine + 1])))

    def _pdp_explanation(self):
        """
        TODO: doc, better prints
        Returns
        -------

        """
        pdp_df = pd.DataFrame(self.x_test)
        pdp_sex = pdp.pdp_isolate(model=self.biased_model,
                                  dataset=pdp_df,
                                  model_features=pdp_df.columns.tolist(),
                                  feature=pdp_df.columns.tolist()[self.biased_id],
                                  num_grid_points=100
                                  )
        fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.input_feature_names[self.biased_id],
                                  plot_lines=True)
        plt.show()
        pdp_sex = pdp.pdp_isolate(model=self.adversarial_model,
                                  dataset=pdp_df,
                                  model_features=pdp_df.columns.tolist(),
                                  feature=pdp_df.columns.tolist()[self.biased_id],
                                  num_grid_points=100
                                  )
        fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.input_feature_names[self.biased_id],
                                  plot_lines=True)
        plt.show()
        print("Prediction fidelity: {0:3.2}".format(
            self.adversarial_model.fidelity(pdp_df.to_numpy())))

    def _lime_explanation(self):
        """
        TODO: doc better prints
        Returns
        -------

        """
        ex_indc = np.random.choice(self.x_test.shape[0])
        if self.ml_type == MLType.CLASSIFICATION_BINARY:
            normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                                      feature_names=self.input_feature_names,
                                                                      discretize_continuous=False,
                                                                      categorical_features=self.categorical_feature_indices,
                                                                      random_state=self.seed)

            normal_exp = normal_explainer.explain_instance(self.x_test[ex_indc],
                                                           self.biased_model.predict_proba,
                                                           num_features=len(self.x_train[1])).as_list()
        elif self.ml_type == MLType.REGRESSION:
            normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                                      feature_names=self.input_feature_names,
                                                                      discretize_continuous=False,
                                                                      categorical_features=self.categorical_feature_indices,
                                                                      mode='regression',
                                                                      random_state=self.seed)

            normal_exp = normal_explainer.explain_instance(self.x_test[ex_indc],
                                                           self.biased_model.predict,
                                                           num_features=len(self.x_train[1])).as_list()
        else:
            raise ValueError()
        print("Explanation on biased f:\n", normal_exp, "\n\n")
        if self.ml_type == MLType.CLASSIFICATION_BINARY:

            adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                                   feature_names=self.input_feature_names,
                                                                   discretize_continuous=False,
                                                                   categorical_features=self.categorical_feature_indices,
                                                                   random_state=self.seed)

            adv_exp = adv_explainer.explain_instance(self.x_test[ex_indc],
                                                     self.adversarial_model.predict_proba,
                                                     num_features=len(self.x_train[1])).as_list()
        elif self.ml_type == MLType.REGRESSION:
            adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train,
                                                                   feature_names=self.input_feature_names,
                                                                   discretize_continuous=False,
                                                                   categorical_features=self.categorical_feature_indices,
                                                                   mode='regression',
                                                                   random_state=self.seed)

            adv_exp = adv_explainer.explain_instance(self.x_test[ex_indc],
                                                     self.adversarial_model.predict,
                                                     num_features=len(self.x_train[1])).as_list()
        else:
            raise ValueError()
        print("Explanation on adversarial model:\n", adv_exp, "\n")
        print("Prediction fidelity: {0:3.2}".format(
            self.adversarial_model.fidelity(self.x_test[ex_indc:ex_indc + 1])))
