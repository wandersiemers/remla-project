from abc import abstractmethod
from joblib import dump


class BaseModel:

    def __init__(self, logging=True):
        self.logging = logging

    @abstractmethod
    def get_features(self, X):
        raise NotImplementedError

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    def save(self, path: str):
        dump(self, path)

    def set_logging(self, flag: bool):
        self.logging = flag
