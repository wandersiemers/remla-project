import os
import re
from ast import literal_eval

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile(r"[^0-9a-z #+_]")
STOP_WORDS = set(stopwords.words("english"))


def read_data(filename: str):
    data = pd.read_csv(filename, sep="\t")
    data["tags"] = data["tags"].apply(literal_eval)

    return data


def text_prepare(text: str):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])

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


def main():
    train = read_data("assets/data/raw/train.tsv")
    validation = read_data("assets/data/raw/validation.tsv")
    test = pd.read_csv("assets/data/raw/test.tsv", sep="\t")

    X_train, y_train = train["title"].values, train["tags"].values
    X_val, y_val = validation["title"].values, validation["tags"].values
    X_test = test["title"].values

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    os.makedirs(os.path.join("assets", "data", "processed"), exist_ok=True)

    write_to_file(X_train, y_train, "train.tsv")
    write_to_file(X_val, y_val, "validation.tsv")
    write_to_file(X_test)


if __name__ == "__main__":
    main()
