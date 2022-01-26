import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from adversarial_model_toolbox import AdversarialModelToolbox
from util.explainer_type import ExplainerType
from util.ml_type import MLType

TRAIN_TEST_SPLIT = 0.2
SEED = 666

dataset_name = 'blobs'
input_ids = [0, 1]
categorical_input_ids = []
output_id = 3
biased_id = 1

# Load dataset
blob_df = pd.read_csv(os.path.join("datasets", dataset_name + ".csv"))
print(blob_df.columns.tolist())

# Split input and output
y_df = blob_df.iloc[:, output_id]
x_df = blob_df.iloc[:, input_ids]

scaler = MinMaxScaler()
scaler.fit(x_df.to_numpy())
x = scaler.transform(x_df.to_numpy())

# Create train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y_df.to_numpy(), test_size=TRAIN_TEST_SPLIT, random_state=SEED)

# Create training data without biased column
ux_train = x_train.copy()
ux_train[:, input_ids.index(biased_id)] = 0
ux_test = x_test.copy()
ux_test[:, input_ids.index(biased_id)] = 0

biased_ml = RandomForestClassifier(random_state=SEED)
biased_ml.fit(x_train, y_train)
print(biased_ml.score(x_test, y_test))

unbiased_ml = RandomForestClassifier(random_state=SEED)
unbiased_ml.fit(ux_test, y_test)
print(unbiased_ml.score(ux_test, y_test))

adv = AdversarialModelToolbox(biased_model=biased_ml, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                              input_feature_names=blob_df.columns[input_ids].tolist(),
                              categorical_feature_indices=[input_ids.index(cat_id) for cat_id in categorical_input_ids],
                              unbiased_model=unbiased_ml,
                              biased_id=input_ids.index(biased_id), fool_explainer_type=ExplainerType.PDP,
                              ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)
adv.train()
adv.get_explanations()
