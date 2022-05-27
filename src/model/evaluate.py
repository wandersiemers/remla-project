import json
import os
from typing import Dict

import numpy as np
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def print_evaluation_scores(y_val: np.ndarray, predicted: np.ndarray):
    print("Accuracy score: ", accuracy_score(y_val, predicted))
    print("F1 score: ", f1_score(y_val, predicted, average="weighted"))
    print(
        "Average precision score: ",
        average_precision_score(y_val, predicted, average="macro"),
    )


def save_evaluation_scores(y_val: np.ndarray, predicted: np.ndarray, algorithm: str):
    res = {
        "Accuracy": accuracy_score(y_val, predicted),
        "F1 score": f1_score(y_val, predicted, average="weighted"),
        "Average precision score": average_precision_score(
            y_val, predicted, average="macro"
        ),
    }
    with open(f"assets/metrics/{algorithm}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f)


def print_words_for_tag(
    classifier: BaseEstimator,
    tag: str,
    tags_classes: list,
    index_to_words: Dict[int, str],
):
    """
    Print top 5 positive and top 5 negative words for current tag

    Parameters
    ---------
    classifier
            trained classifier
    tag
            a particular tag
    tags_classes
            list of classes names from MultiLabelBinarizer
    index_to_words
            index_to_words transformation
    all_words
            all words in the dictionary
    """

    print("fTag:\t{tag}")

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]
    ]
    top_negative_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]
    ]

    print(f"Top positive words:\t{', '.join(top_positive_words)}")
    print(f"Top negative words:\t{', '.join(top_negative_words)}\n")


def main():
    y_val = load("assets/outputs/y_val.joblib")
    y_val_predicted_labels_mybag = load("assets/outputs/y_val_predicted_mybag.joblib")
    y_val_predicted_labels_tfidf = load("assets/outputs/y_val_predicted_tfidf.joblib")
    tfidf_reversed_vocab = load("assets/outputs/tf_idf_reversed_vocab.joblib")
    classifier_tfidf = load("assets/models/classifier_tfidf.joblib")
    mlb_classifier = load("assets/models/mlb_classifier.joblib")

    os.makedirs(os.path.join("assets", "metrics"), exist_ok=True)

    print("Bag-of-words")
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    save_evaluation_scores(y_val, y_val_predicted_labels_mybag, "bag-of-words")
    print("Tfidf")
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
    save_evaluation_scores(y_val, y_val_predicted_labels_tfidf, "tf-idf")

    print_words_for_tag(
        classifier_tfidf, "c", mlb_classifier.classes, tfidf_reversed_vocab
    )
    print_words_for_tag(
        classifier_tfidf, "c++", mlb_classifier.classes, tfidf_reversed_vocab
    )
    print_words_for_tag(
        classifier_tfidf, "linux", mlb_classifier.classes, tfidf_reversed_vocab
    )


if __name__ == "__main__":
    main()
