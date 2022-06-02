import os
from typing import Dict, List, Tuple, Type

from data.preprocessing import read_files
from model.bag_model import BagModel
from model.base_model import BaseModel
from model.tfidf_model import TfIdfModel


def get_corpus_counts(X_train: list[str], y_train: list[str]):
    tags_counts: Dict[str, int] = {}
    words_counts: Dict[str, int] = {}

    for sentence in X_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    return words_counts, tags_counts


def main():
    X_train, y_train, _, _, _ = read_files("processed")

    words_counts, tags_counts = get_corpus_counts(X_train, y_train)

    # Make sure that we can save the models
    os.makedirs(os.path.join("assets", "models"), exist_ok=True)

    base_config = {"classes": tags_counts.keys(), "words_counts": words_counts}

    models: List[Tuple[str, Type[BaseModel], Dict]] = [
        ("tfidf_classifier", TfIdfModel, base_config),
        ("bag_classifier", BagModel, base_config),
    ]

    for classifier_name, Model, model_config in models:
        model = Model(True, model_config)

        model.train(model.get_features(X_train), model.get_labels(y_train))
        model.save(f"assets/models/{classifier_name}.joblib")


if __name__ == "__main__":
    main()
