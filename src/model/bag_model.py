from typing import Any, Dict

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .base_model import BaseModel

DEFAULT_BAG_MODEL_CONFIG = {"C": 1, "penalty": "l1", "dict_size": 5000}


def _bag_of_words(text: str, words_to_index: Dict[str, int], dict_size: int):
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1

    return result_vector


def _sparse_bag_of_words(
    X: list[str], words_to_index: Dict[str, int], dict_size: int
) -> sp_sparse.bmat:
    return sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(_bag_of_words(text, words_to_index, dict_size))
            for text in X
        ]
    )


class BagModel(BaseModel):
    def __init__(self, logging: bool, config: Dict[str, Any]):
        BaseModel.__init__(self, logging)

        self._words_counts = config["words_counts"]
        self._dict_size = config.get("dict_size", DEFAULT_BAG_MODEL_CONFIG["dict_size"])

        self._mlb = MultiLabelBinarizer(classes=sorted(config["classes"]))

        # Initialize the bag of words
        self._index_to_words: list[str] = sorted(
            self._words_counts, key=self._words_counts.get, reverse=True  # type: ignore
        )[: self._dict_size]
        self._words_to_index = {word: i for i, word in enumerate(self._index_to_words)}

        # Initialize the classifier
        penalty: str = config.get("penalty", DEFAULT_BAG_MODEL_CONFIG["penalty"])
        C: int = config.get("C", DEFAULT_BAG_MODEL_CONFIG["C"])
        clf = LogisticRegression(
            penalty=penalty, C=C, dual=False, solver="liblinear", verbose=1
        )
        self._classifier = OneVsRestClassifier(clf, verbose=1)

    def get_features(self, X: list[str]):
        return _sparse_bag_of_words(X, self._words_to_index, self._dict_size)

    def get_labels(self, y: list[str]):
        return self._mlb.fit_transform(y)

    def train(self, X_train: sp_sparse.bmat, y_train: sp_sparse.csr_matrix):
        self._classifier.fit(X_train, y_train)

    def predict(self, X_test: list[str]):
        X_featurized = self.get_features(X_test)

        return self._classifier.predict(X_featurized)
