from abc import abstractmethod
from typing import Generic, TypeVar

from joblib import dump

T = TypeVar("T")
U = TypeVar("U")


class BaseModel(Generic[T, U]):
    def __init__(self, logging: bool = True):
        self.logging = logging

    @abstractmethod
    def get_features(self, X: T):
        raise NotImplementedError

    @abstractmethod
    def get_labels(self, y: U):
        raise NotImplementedError

    @abstractmethod
    def train(self, X_train: T, y_train: U):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: T) -> U:
        raise NotImplementedError

    def save(self, path: str):
        dump(self, path)

    def set_logging(self, flag: bool):
        self.logging = flag
