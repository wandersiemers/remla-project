from typing import Any, Dict

from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from remla.models.base_model import BaseModel

DEFAULT_TF_IDF_MODEL_CONFIG = {"C": 1, "penalty": "l1", "dict_size": 5000}


class TfIdfModel(BaseModel):
    def __init__(self, logging: bool, config: Dict[str, Any]):
        BaseModel.__init__(self, logging)

        self._mlb = MultiLabelBinarizer(classes=sorted(config["classes"]))

        self._tfidf_vectorizer = TfidfVectorizer(
            min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r"(\S+)"
        )
        vocabulary: Dict[str, int] = self._tfidf_vectorizer.vocabulary_
        self.tfidf_reversed_vocab = {i: word for word, i in vocabulary.items()}

        # Initialize the classifier
        penalty: str = config.get("penalty", DEFAULT_TF_IDF_MODEL_CONFIG["penalty"])
        C: int = config.get("C", DEFAULT_TF_IDF_MODEL_CONFIG["C"])
        clf = LogisticRegression(
            penalty=penalty, C=C, dual=False, solver="liblinear", verbose=1
        )
        self._classifier = OneVsRestClassifier(clf, verbose=1)

    def get_features(self, X: list[str]):
        return self._tfidf_vectorizer.fit_transform(X)

    def get_labels(self, y: list[str]):
        return self._mlb.fit_transform(y)

    def train(self, X_train: sp_sparse.csr_matrix, y_train: sp_sparse.csr_matrix):
        self._classifier.fit(X_train, y_train)

    def predict(self, X_test: list[str]):
        X_featurized = self.get_features(X_test)

        return self._classifier.predict(X_featurized)
