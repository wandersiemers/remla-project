from src.model.base_model import BaseModel
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from scipy import sparse as sp_sparse


def _bag_of_words(text: str, words_to_index: Dict[str, int], dict_size: int):
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1

    return result_vector


class MyBagModel(BaseModel):

    def __init__(self, logging: bool, config: Dict[str, Any]):
        BaseModel.__init__(self, logging)

        self.mlb = MultiLabelBinarizer(classes=sorted(config["classes"]))

        self.words_counts = config.get("words_counts")
        self.dict_size = config.get("dict_size", 5000)

        self.index_to_words: list[str] = sorted(
            self.words_counts, key=self.words_counts.get, reverse=True  # type: ignore
        )[:self.dict_size]

        penalty = config.get("penalty", "l1")
        C = config.get("C", 1)
        clf = LogisticRegression(
            penalty=penalty, C=C, dual=False, solver="liblinear", verbose=1
        )
        self.classifier = OneVsRestClassifier(clf, verbose=1)

    def _sparse_bag_of_words(self, X: list[str], words_to_index: Dict[str, int]) -> sp_sparse.bmat:
        return sp_sparse.vstack(
            [
                sp_sparse.csr_matrix(_bag_of_words(text, words_to_index, self.dict_size))
                for text in X
            ]
        )

    def get_features(self, X):
        words_to_index = {word: i for i, word in enumerate(self.index_to_words)}
        X_bag = self._sparse_bag_of_words(X, words_to_index)
        return X_bag

    def get_labels(self, y):
        return self.mlb.fit_transform(y)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        featurized = self.get_features(X_test)
        return self.classifier.predict(featurized)
