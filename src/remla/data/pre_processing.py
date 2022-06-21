import os
import re
from ast import literal_eval

import nltk
import numpy as np
import pandas as pd
import wandb
from nltk.corpus import stopwords

from remla.config import wandb_entity, wandb_project_name

nltk.download("stopwords")


def read_data(filename: str):
    data = pd.read_csv(filename, sep="\t")
    data["tags"] = data["tags"].apply(literal_eval)

    return data


def text_prepare(text: str):
    replace_by_space_re = re.compile(r"[/(){}\[\]\|@,;]")
    bad_symbols_re = re.compile(r"[^0-9a-z #+_]")
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(replace_by_space_re, " ", text)
    text = re.sub(bad_symbols_re, "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def write_to_file(X: list[str], y: np.ndarray = None, filename: str = "test.tsv"):
    assert y is None or len(X) == len(
        y
    ), "Number of feature rows should equal number of label lists"

    dataframe = (
        pd.DataFrame({"title": X})
        if y is None
        else pd.DataFrame({"title": X, "tags": y})
    )

    dataframe.to_csv(f"assets/data/processed/{filename}", sep="\t", index=False)


def read_files(directory: str):
    train = read_data(f"assets/data/{directory}/train.tsv")
    validation = read_data(f"assets/data/{directory}/validation.tsv")
    test = pd.read_csv(f"assets/data/{directory}/test.tsv", sep="\t")

    X_train, y_train = train["title"].values, train["tags"].values
    X_val, y_val = validation["title"].values, validation["tags"].values
    X_test = test["title"].values

    return X_train, y_train, X_val, y_val, X_test


def main():
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        tags=["data"],
    )

    X_train, y_train, X_val, y_val, X_test = read_files("raw")

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    os.makedirs(os.path.join("assets", "data", "processed"), exist_ok=True)

    write_to_file(X_train, y_train, "train.tsv")
    write_to_file(X_val, y_val, "validation.tsv")
    write_to_file(X_test)

    y_train_labels = np.concatenate(y_train)
    y_val_labels = np.concatenate(y_val)
    classes = set(np.concatenate((y_train_labels, y_val_labels)))

    wandb.sklearn.plot_class_proportions(y_train_labels, y_val_labels, classes)

    wandb.finish()


if __name__ == "__main__":
    main()
