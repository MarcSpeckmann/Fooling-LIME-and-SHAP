from sklearn.ensemble import RandomForestClassifier

from adversarial_model_toolbox import AdversarialModelToolbox
from dataset import Dataset
from util.explainer_type import ExplainerType
from util.ml_type import MLType

TRAIN_TEST_SPLIT = 0.8
SEED = 666

dataset = Dataset(dataset_name='blobs', output_id=3, input_ids=list(range(3)),
                  categorical_ids=[], biased_id=1, normalize=True, split=TRAIN_TEST_SPLIT, random_state=SEED)

(X_train, y_train), (X_test, y_test) = dataset.get_data(biased=True)
biased_ml = RandomForestClassifier(max_depth=100, random_state=SEED)
biased_ml.fit(X_train, y_train)

(uX_train, uy_train), (uX_test, uy_test) = dataset.get_data(biased=False)
unbiased_ml = RandomForestClassifier(max_depth=100, random_state=SEED)
unbiased_ml.fit(uX_train, uy_train)

adv = AdversarialModelToolbox(biased_model=biased_ml, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test,
                              input_feature_names=dataset.get_input_labels(),
                              categorical_feature_indices=dataset.categorical_ids, unbiased_model=unbiased_ml,
                              biased_id=dataset.biased_id, fool_explainer_type=ExplainerType.PDP,
                              ml_type=MLType.CLASSIFICATION_BINARY, seed=SEED)

adv.train()
adv.get_explanations()
