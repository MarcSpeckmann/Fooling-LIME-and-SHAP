import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from adversarial_model_toolbox import AdversarialModelToolbox
from dataset import Dataset
from util.explainer_type import ExplainerType
from util.ml_type import MLType

TRAIN_TEST_SPLIT = 0.8
SEED = 666

dataset = Dataset(dataset_name='HousingData', output_id=13, input_ids=list(range(13)),
                  categorical_ids=[], biased_ids=[12], normalize=True)
#dataset = Dataset(dataset_name='blobs', output_id=3, input_ids=[0,1,2],
#                  categorical_ids=[3], biased_ids=[1], normalize=True)

#dataset = Dataset(dataset_name='blob_3_class', output_id=4, input_ids=[0,1,2,3],
#                  categorical_ids=[4], biased_ids=[3], normalize=True)

(X_train, y_train), (X_test, y_test) = dataset.get_data(biased=True, split=TRAIN_TEST_SPLIT, random_state=SEED)
biased_ml = RandomForestRegressor(max_depth=10, random_state=SEED)
biased_ml.fit(X_train, y_train)

(uX_train, uy_train), (uX_test, uy_test) = dataset.get_data(biased=False, split=TRAIN_TEST_SPLIT, random_state=SEED)
unbiased_ml = RandomForestRegressor(max_depth=10, random_state=SEED)
unbiased_ml.fit(uX_train, uy_train)


adv = AdversarialModelToolbox(biased_model=biased_ml, data=dataset, unbiased_model=unbiased_ml,
                              fool_explainer_type=ExplainerType.SHAP, ml_type=MLType.CLASSIFICATION, train_test_split=TRAIN_TEST_SPLIT, seed=SEED)
adv.train()
adv.get_explanations()
