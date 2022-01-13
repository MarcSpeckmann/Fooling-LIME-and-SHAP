import os
from typing import List

import numpy as np
import pandas as pd
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
                 biased_ids: List[int],
                 categorical_ids: List[int] = None,
                 normalize: bool = False,
                 impute_strategy: str = "remove"):
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

        self.biased_id = biased_ids
        if categorical_ids:
            for categorical_id in categorical_ids:
                self._data[self._data.columns[categorical_id]] = pd.Categorical(
                    self._data[self._data.columns[categorical_id]])

        self.dataset_name = dataset_name
        self.input_ids = input_ids
        self.categorical_ids = categorical_ids

        if isinstance(output_id, list) and len(output_id) == 1:
            output_id = output_id[0]

        self.output_id = output_id

        X, y = self._data.to_numpy()[:, self.input_ids], self._data.to_numpy()[:, self.output_id].reshape(-1, 1)

        if impute_strategy is not None:
            # Set NaN values to 0
            if impute_strategy == "zeros":
                X[pd.isnull(X)] = 0
            elif impute_strategy == "remove":
                # First take care of X
                mask = np.any(pd.isnull(X), axis=1)
                X = X[~mask]
                y = y[~mask]

                # Then take care of y
                mask = np.any(pd.isnull(y), axis=1)
                X = X[~mask]
                y = y[~mask]
            else:
                raise NotImplementedError("Impute strategy was not found.")

        if normalize:
            scaler = MinMaxScaler()

            scaler.fit(X)
            X = scaler.transform(X)

            if 'category' != str(self._data.iloc[:, self.output_id].dtype):
                scaler.fit(y)
                y = scaler.transform(y)

        self.X = X
        self.y = y

    def get_data(self, biased=True, split=0.6, random_state=0):
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
        X, y = shuffle(self.X, self.y, random_state=random_state)
        if biased == False:
            X[:, self.biased_id] = 0

        split_idx = int(len(X) * split)

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        y_train = y_train.flatten()
        y_val = y_val.flatten()

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
