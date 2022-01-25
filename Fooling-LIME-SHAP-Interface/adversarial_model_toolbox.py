import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pdpbox import pdp, info_plots
from sklearn.inspection import PartialDependenceDisplay

from adversarial_kernel_shap_model import AdversarialKernelSHAPModel
from adversarial_pdp_model import AdversarialPDPModel
from adversarial_lime_model import AdversarialLimeModel
from dataset import Dataset
from util.explainer_type import ExplainerType
from util.ml_type import MLType


class AdversarialModelToolbox:
    """
    TODO:
    """

    def __init__(self, biased_model, data: Dataset, unbiased_model=None,
                 fool_explainer_type: ExplainerType = ExplainerType.LIME, ml_type: MLType = MLType.CLASSIFICATION_BINARY,
                 seed: int = 0):
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
        self.ml_type = ml_type

        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.data.get_data()

        if fool_explainer_type == ExplainerType.LIME :
            self.adversarial_model = AdversarialLimeModel(self.biased_model, self.unbiased_model)
        elif fool_explainer_type == ExplainerType.SHAP:
            self.adversarial_model = AdversarialKernelSHAPModel(self.biased_model, self.unbiased_model)
        elif fool_explainer_type == ExplainerType.PDP:
            self.adversarial_model = AdversarialPDPModel(self.biased_model, self.unbiased_model)
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
        elif self.type == ExplainerType.PDP:
            self.adversarial_model.train(self.X_train, hide_index=self.data.biased_id, feature_names=self.data.get_input_labels(),
                                         categorical_features=self.data.get_input_categorical_feature_indices(),
                                         estimator=None)

    def get_explanations(self):
        """
        TODO:
        Returns
        -------

        """
        if self.type == ExplainerType.LIME:
            ex_indc = np.random.choice(self.X_test.shape[0])
            if self.ml_type == MLType.CLASSIFICATION_BINARY:
                normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                          feature_names=self.adversarial_model.get_column_names(),
                                                                          discretize_continuous=False,
                                                                          categorical_features=self.data.get_input_categorical_feature_indices())

                normal_exp = normal_explainer.explain_instance(self.X_test[ex_indc],
                                                               self.biased_model.predict_proba,
                                                               num_features=len(self.X_train[1])).as_list()
            elif self.ml_type == MLType.REGRESSION:
                normal_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                          feature_names=self.adversarial_model.get_column_names(),
                                                                          discretize_continuous=False,
                                                                          categorical_features=self.data.get_input_categorical_feature_indices(),
                                                                          mode='regression')

                normal_exp = normal_explainer.explain_instance(self.X_test[ex_indc],
                                                               self.biased_model.predict,
                                                               num_features=len(self.X_train[1])).as_list()
            else:
                raise ValueError()

            print("Explanation on biased f:\n", normal_exp, "\n\n")

            if self.ml_type == MLType.CLASSIFICATION_BINARY:

                adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                       feature_names=self.adversarial_model.get_column_names(),
                                                                       discretize_continuous=False,
                                                                       categorical_features=self.data.get_input_categorical_feature_indices())

                adv_exp = adv_explainer.explain_instance(self.X_test[ex_indc],
                                                         self.adversarial_model.predict_proba,
                                                         num_features=len(self.X_train[1])).as_list()
            elif self.ml_type == MLType.REGRESSION:
                adv_explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,
                                                                       feature_names=self.adversarial_model.get_column_names(),
                                                                       discretize_continuous=False,
                                                                       categorical_features=self.data.get_input_categorical_feature_indices(),
                                                                       mode='regression')

                adv_exp = adv_explainer.explain_instance(self.X_test[ex_indc],
                                                         self.adversarial_model.predict,
                                                         num_features=len(self.X_train[1])).as_list()
            else:
                raise ValueError()

            print("Explanation on adversarial model:\n", adv_exp, "\n")

            print("Prediction fidelity: {0:3.2}".format(
                self.adversarial_model.fidelity(self.X_test[ex_indc:ex_indc + 1])))
        elif self.type == ExplainerType.SHAP:
            # TODO: check for different ml types
            background_distribution = shap.kmeans(self.X_train, 10)

            to_examine = np.random.choice(self.X_test.shape[0])

            biased_kernel_explainer = shap.KernelExplainer(self.biased_model.predict, background_distribution)
            biased_shap_values = biased_kernel_explainer.shap_values(self.X_test[to_examine:to_examine + 1])

            adv_kerenel_explainer = shap.KernelExplainer(self.adversarial_model.predict, background_distribution)
            adv_shap_values = adv_kerenel_explainer.shap_values(self.X_test[to_examine:to_examine + 1])

            shap.summary_plot(biased_shap_values, feature_names=self.data.get_input_labels(), plot_type="bar")
            shap.summary_plot(adv_shap_values, feature_names=self.data.get_input_labels(), plot_type="bar")

            print("Fidelity: {0:3.2}".format(self.adversarial_model.fidelity(self.X_test[to_examine:to_examine + 1])))

        elif self.type == ExplainerType.PDP:
            (X_train, y_train), (X_test, y_test) = self.data.get_data()
            pdp_df = pd.DataFrame(X_test)
            pdp_sex = pdp.pdp_isolate(model=self.biased_model,
                                      dataset=pdp_df,
                                      model_features=pdp_df.columns.tolist(),
                                      feature=pdp_df.columns.tolist()[self.data.biased_id],
                                      num_grid_points=100
                                      )
            fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.data.get_biased_label(), plot_lines=True)
            plt.show()

            pdp_sex = pdp.pdp_isolate(model=self.adversarial_model,
                                      dataset=pdp_df,
                                      model_features=pdp_df.columns.tolist(),
                                      feature=pdp_df.columns.tolist()[self.data.biased_id],
                                      num_grid_points=100
                                      )
            fig, axes, = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name=self.data.get_biased_label(),
                                      plot_lines=True)
            plt.show()

