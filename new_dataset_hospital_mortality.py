import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from adversarial_model_toolbox import AdversarialModelToolbox
from util.explainer_type import ExplainerType
from util.ml_type import MLType

TRAIN_TEST_SPLIT = 0.2
SEED = 666

dataset_name = 'data01'
input_ids = list(range(3, 10))
categorical_input_ids = [4, 6, 7, 8, 9]
output_id = 2
biased_id = 4
categorical_ids_in_input = [input_ids.index(cat_id) for cat_id in categorical_input_ids]

# Load dataset
mortality_df = pd.read_csv(os.path.join("datasets", dataset_name + ".csv"))
mortality_df = mortality_df.fillna(0)

# Split input and output
y_df = mortality_df.iloc[:, output_id]
x_df = mortality_df.iloc[:, input_ids]

scaler = MinMaxScaler()
scaler.fit(x_df.to_numpy())
x = scaler.transform(x_df.to_numpy())

for idx in categorical_ids_in_input:
    x[idx] = x_df.iloc[idx]
    le = LabelEncoder()
    x[idx] = le.fit_transform(x[idx])

# Create train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y_df.to_numpy(), test_size=TRAIN_TEST_SPLIT, random_state=SEED)

# Create training data without biased column
ux_train = x_train.copy()
ux_train[:, input_ids.index(biased_id)] = 0
ux_test = x_test.copy()
ux_test[:, input_ids.index(biased_id)] = 0

biased_ml = RandomForestClassifier(random_state=SEED)
biased_ml.fit(x_train, y_train)
print("Accuracy of biased model: {0:3.2}".format(biased_ml.score(x_test, y_test)))

unbiased_ml = RandomForestClassifier(random_state=SEED)
unbiased_ml.fit(ux_test, y_test)
print("Accuracy of unbiased model: {0:3.2}".format(unbiased_ml.score(ux_test, y_test)))

adv = AdversarialModelToolbox(biased_model=biased_ml, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                              input_feature_names=mortality_df.columns[input_ids].tolist(),
                              categorical_feature_indices=categorical_ids_in_input,
                              unbiased_model=unbiased_ml,
                              biased_id=input_ids.index(biased_id), fool_explainer_type=ExplainerType.LIME,
                              ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)
adv.train()
adv.get_explanations()

adv = AdversarialModelToolbox(biased_model=biased_ml, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                              input_feature_names=mortality_df.columns[input_ids].tolist(),
                              categorical_feature_indices=categorical_ids_in_input,
                              unbiased_model=unbiased_ml,
                              biased_id=input_ids.index(biased_id), fool_explainer_type=ExplainerType.SHAP,
                              ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)
adv.train()
adv.get_explanations()
