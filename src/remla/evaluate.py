import argparse
from typing import Dict, Type

import numpy as np
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

import wandb
from remla.config import wandb_entity, wandb_project_name
from remla.data.pre_processing import read_files
from remla.models.base_model import BaseModel


def log_evaluation_scores(
    y_val: np.ndarray, predicted: np.ndarray, classifier_name: str
):
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        tags=["model"],
        config={"classifier": classifier_name},
    )

    accuracy = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average="weighted")
    average_precision = average_precision_score(y_val, predicted, average="macro")

    wandb.log(
        {
            "accuracy": accuracy,
            "f1": f1,
            "average_precision": average_precision,
        }
    )

    print(classifier_name)
    print("Accuracy score: ", accuracy)
    print("F1 score: ", f1)
    print("Average precision score: ", average_precision)


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

    print(f"Tag:\t{tag}")

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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained machine learning model on processed data from the data pipeline"
    )
    parser.add_argument(
        "-jfp",
        "--joblib-file-path",
        help="Path to Joblib file containing the model class",
        required=True,
    )
    args = parser.parse_args()

    joblib_file_path: str = args.joblib_file_path

    _, _, X_val, y_val, _ = read_files("processed")

    classifier: Type[BaseModel] = load(joblib_file_path)

    classifier_name: str = classifier.__class__.__name__

    log_evaluation_scores(
        classifier.get_labels(y_val), classifier.predict(X_val), classifier_name
    )

    if classifier_name == "TfIdfModel":
        for tag in ["c", "c++", "linux"]:
            print_words_for_tag(
                classifier._classifier,
                tag,
                classifier._mlb.classes,
                classifier.tfidf_reversed_vocab,
            )


if __name__ == "__main__":
    main()
