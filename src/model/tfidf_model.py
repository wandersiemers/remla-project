from src.model.base_model import BaseModel
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load


class TfidfModel(BaseModel):

    def __init__(self, logging: bool, config: Dict[str, Any]):
        BaseModel.__init__(self, logging)
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r"(\S+)"
        )
        self.mlb = MultiLabelBinarizer(classes=sorted(config["classes"]))

        penalty = config.get("penalty", "l1")
        C = config.get("C", 1)
        clf = LogisticRegression(
            penalty=penalty, C=C, dual=False, solver="liblinear", verbose=1
        )
        self.classifier = OneVsRestClassifier(clf, verbose=1)

    def get_features(self, X):
        X_train = self.tfidf_vectorizer.fit_transform(X)
        return X_train

    def get_labels(self, y):
        return self.mlb.fit_transform(y)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        featurized = self.get_features(X_test)
        return self.classifier.predict(featurized)

    def save(self, path):
        dump(self, path)
