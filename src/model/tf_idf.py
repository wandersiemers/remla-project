from typing import Dict
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(X_train: list[str], X_val: list[str], X_test: list[str]):
    """
    Parameters
    ----------
    X_train, X_val, X_test — samples

    Returns
    -------
    TF-IDF-vectorized representation of each sample and vocabulary
    """

    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern=r"(\S+)"
    )

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    vocabulary: Dict[str, int] = tfidf_vectorizer.vocabulary_

    dump(tfidf_vectorizer, 'assets/outputs/tfidf-vectorizer.joblib')

    return X_train, X_val, X_test, vocabulary
