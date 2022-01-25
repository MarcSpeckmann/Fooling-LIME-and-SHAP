import os
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


class Dataset:
    """
    TODO:
    """

    def __init__(self,
                 dataset_name: str,
                 input_ids: List[int],
                 output_id: int,
                 biased_id: int,
                 categorical_ids: List[int] = None,
                 normalize: bool = False,
                 split=0.6, random_state=0):
        """
        TODO:
        Parameters
        ----------
        dataset_name
        input_ids
        output_id
        categorical_ids
        normalize
        impute_strategy
        """

        self._data = pd.read_csv(os.path.join("datasets", dataset_name + ".csv"))

        self.biased_raw = biased_id
        self.biased_id = input_ids.index(biased_id)

        self.label_encoder = []
        if categorical_ids:
            for categorical_id in categorical_ids:
                self._data[self._data.columns[categorical_id]] = pd.Categorical(self._data[self._data.columns[categorical_id]])
                le = preprocessing.LabelEncoder()
                self._data[self._data.columns[categorical_id]] = le.fit_transform(self._data[self._data.columns[categorical_id]])
                self.label_encoder.append(le)


        self.dataset_name = dataset_name
        self.input_ids = input_ids
        self.categorical_ids = categorical_ids

        if isinstance(output_id, list) and len(output_id) == 1:
            output_id = output_id[0]

        self.output_id = output_id

        self._data = self._data.fillna(0)

        X, y = self._data.to_numpy()[:, self.input_ids], self._data.to_numpy()[:, self.output_id].reshape(-1, 1)

        if normalize:
            scaler = MinMaxScaler()

            scaler.fit(X)
            X = scaler.transform(X)

            if 'category' != str(self._data.iloc[:, self.output_id].dtype):
                scaler.fit(y)
                y = scaler.transform(y)

        self.X, self.y = shuffle(X, y, random_state=random_state)

        split_idx = int(len(X) * split)

        self.X_train, self.y_train = X[:split_idx], y[:split_idx]
        self.X_val, self.y_val = X[split_idx:], y[split_idx:]

        self.y_train = self.y_train.flatten()
        self.y_val = self.y_val.flatten()

    def get_data(self, biased=True, df=False):
        """
        TODO:
        Parameters
        ----------
        biased
        split
        random_state

        Returns
        -------

        """
        if df:
            X_train = pd.DataFrame(self.X_train)
            X_train.columns = self._data.columns[self.input_ids]
            X_val = pd.DataFrame(self.X_val)
            X_val.columns = self._data.columns[self.input_ids]
            y_train = pd.DataFrame({self._data.columns[self.output_id]: self.y_train})
            y_val = pd.DataFrame({self._data.columns[self.output_id]: self.y_val})
        else:
            y_train = self.y_train.copy()
            y_val = self.y_val.copy()
            X_train = self.X_train.copy()
            X_val = self.X_val.copy()

        if biased == False:
            X_train[:, self.biased_id] = 0
            X_val[:, self.biased_id] = 0

        return (X_train, y_train), (X_val, y_val)

    def get_input_labels(self, idx=None):
        """
        TODO:
        Parameters
        ----------
        idx

        Returns
        -------

        """
        label = self._data.columns.to_numpy()
        if idx is None:
            return label[self.input_ids].tolist()
        else:
            return label[self.input_ids][idx].tolist()

    def get_output_label(self):
        """
        TODO:
        Returns
        -------

        """
        return self._data.columns[self.output_id]

    def get_input_categorical_feature_indices(self):
        """
        TODO:
        Returns
        -------

        """
        idxs = []
        if self.categorical_ids:
            for i, input_idx in enumerate(self.input_ids):
                if input_idx in self.categorical_ids:
                    idxs.append(i)
        return idxs

    def get_biased_label(self):
        return self._data.columns.tolist()[self.biased_raw]
