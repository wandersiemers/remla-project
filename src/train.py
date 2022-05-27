import os
from typing import Dict

import pandas as pd
from joblib import dump

from data.preprocessing import read_data
from model import bag_of_words, mlb, tf_idf


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
    train = read_data("assets/data/processed/train.tsv")
    validation = read_data("assets/data/processed/validation.tsv")
    test = pd.read_csv("assets/data/processed/test.tsv", sep="\t")

    X_train, y_train = train["title"].values, train["tags"].values
    X_val, y_val = validation["title"].values, validation["tags"].values
    X_test = test["title"].values

    words_counts, tags_counts = get_corpus_counts(X_train, y_train)
    X_train_mybag, X_val_mybag = bag_of_words.initialize(
        words_counts, X_train, X_val, X_test
    )

    X_train_tfidf, X_val_tfidf, _, tfidf_vocab = tf_idf.tfidf_features(
        X_train, X_val, X_test
    )
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    mlb_classifier, y_train, y_val = mlb.get_mlb(tags_counts, y_train, y_val)
    classifier_mybag = mlb.train_classifier(X_train_mybag, y_train)
    classifier_tfidf = mlb.train_classifier(X_train_tfidf, y_train)

    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)

    os.makedirs(os.path.join("assets", "outputs"), exist_ok=True)
    os.makedirs(os.path.join("assets", "models"), exist_ok=True)

    dump(y_val, "assets/outputs/y_val.joblib")
    dump(y_val_predicted_labels_mybag, "assets/outputs/y_val_predicted_mybag.joblib")
    dump(y_val_predicted_labels_tfidf, "assets/outputs/y_val_predicted_tfidf.joblib")
    dump(tfidf_reversed_vocab, "assets/outputs/tf_idf_reversed_vocab.joblib")
    dump(classifier_tfidf, "assets/models/classifier_tfidf.joblib")
    dump(mlb_classifier, "assets/models/mlb_classifier.joblib")


if __name__ == "__main__":
    main()
