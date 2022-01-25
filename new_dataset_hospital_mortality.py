from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from adversarial_model_toolbox import AdversarialModelToolbox
from dataset import Dataset
from util.explainer_type import ExplainerType
from util.ml_type import MLType

TRAIN_TEST_SPLIT = 0.8
SEED = 666

dataset = Dataset(dataset_name='data01', output_id=2, input_ids=list(range(3, 10)),
                  categorical_ids=[2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], biased_id=4, normalize=True)


(X_train, y_train), (X_test, y_test) = dataset.get_data(biased=True)
biased_ml = RandomForestClassifier(max_depth=100, random_state=SEED)
biased_ml.fit(X_train, y_train)

(uX_train, uy_train), (uX_test, uy_test) = dataset.get_data(biased=False)
unbiased_ml = RandomForestClassifier(max_depth=100, random_state=SEED)
unbiased_ml.fit(uX_train, uy_train)

adv = AdversarialModelToolbox(biased_model=biased_ml, data=dataset, unbiased_model=unbiased_ml,
                              fool_explainer_type=ExplainerType.LIME, ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)
adv.train()
adv.get_explanations()

adv = AdversarialModelToolbox(biased_model=biased_ml, data=dataset, unbiased_model=unbiased_ml,
                              fool_explainer_type=ExplainerType.SHAP, ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)
adv.train()
adv.get_explanations()
