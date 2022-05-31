import os
from typing import Dict

from joblib import dump

from data.preprocessing import read_files
from model.tfidf_model import TfidfModel
from model.mybag_model import MyBagModel


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
    X_train, y_train, X_val, y_val, X_test = read_files("processed")

    words_counts, tags_counts = get_corpus_counts(X_train, y_train)

    os.makedirs(os.path.join("assets", "outputs"), exist_ok=True)
    os.makedirs(os.path.join("assets", "models"), exist_ok=True)

    config = {
        "classes": tags_counts.keys(),
        "words_counts": words_counts
    }

    # Code for the TF-IDF Model
    tf_idf_model = TfidfModel(True, config)
    X_train_tfidf = tf_idf_model.get_features(X_train)
    y_train = tf_idf_model.get_labels(y_train)
    tf_idf_model.train(X_train_tfidf, y_train)
    y_val_predicted_labels_tfidf = tf_idf_model.predict(X_val)
    tf_idf_model.save("assets/models/classifier_tfidf.joblib")

    # Code for the Bag-of-Words Model
    # mybag_model = MyBagModel(True, config)
    # X_train_mybag = mybag_model.get_features(X_train)
    # y_train = mybag_model.get_labels(y_train)
    # mybag_model.train(X_train_mybag, y_train)
    # y_val_predicted_labels_mybag = mybag_model.predict(X_val)
    # mybag_model.save("assets/models/classifier_mybag.joblib")


if __name__ == "__main__":
    main()
