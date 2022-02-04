# Fooling LIME and SHAP

Post-hoc explanation techniques that rely on input pertubations, such as LIME and SHAP, are not reliable towards systematic errors and underlying biases.
In this project, the scaffolding technique from Slack et al. should be re-implemented, which effectively should hide the biases of any given classifier.
- Paper Reference: https://arxiv.org/abs/1911.02508
- Code Reference: https://github.com/dylan-slack/Fooling-LIME-SHAP


## Installation

1. Clone the repository

    ```bash
    git clone https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon.git
    ```

2. Create the environment

    ```bash
    cd iml-ws21-projects-fool_the_lemon
    conda env create -f "environment.yml"
    conda activate iML-project
    ```

3. Run experiments

    To run the experiments notebooks start a jupyterlab server.

    How to install jupyterlab: https://github.com/jupyterlab/jupyterlab

    ```bash
    jupyter-lab .
    ```




## Experiments 

### Reproduction (10)
Implement the approach by writting a simple interface/framework and confirm yiur implementation by using any (tabular) raciscm dataset (e.g. Boston Housing)
- [Reproduction with boston housing dataset](https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon/blob/main/repoduction_with_boston_housing.ipynb)

### Extension (10)

Additionally to LIME and SHAP, incoporate PDP and analyse if it is fool-able, too.
- [PDP with boston housing dataset](https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon/blob/main/fool_pdp_with_boston_housing.ipynb)

### Analysis (5)
Use different perturbation approaches and compare the impact on being fooled.
- [Different perturbation approaches with boston housing dataset](https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon/blob/main/compare_pertubation_approaches_with_boston_housing.ipynb)

### Hyperparameter Sensitivity (10)
Analyze the impact of the hyperparameters of LIME and SHAP (e.g., hyperparameters of the local model and of the pertubation algorithms).

### New Datasets (5)

Find at least two further (tabular) datasets with a risk of discrimination (that are not mentioned in the paper and study the impact of fooling on them.
- [Gender discrimination dataset](https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon/blob/main/new_dataset_gender_discrimination.ipynb)
- [Heart failure prediction dataset](https://github.com/automl-classroom/iml-ws21-projects-fool_the_lemon/blob/main/new_dataset_heart_failure.ipynb)


## Datasets

- [Boson housing dataset](https://www.kaggle.com/altavish/boston-housing-dataset)
- [Gender discrimination dataset](https://www.kaggle.com/hjmjerry/gender-discrimination)
- [Heart failure prediction dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

## Limitations

The current framework can only deal with regression and binary classification tasks