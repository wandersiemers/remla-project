from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def get_mlb(tags_counts, y_train, y_val):
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    return mlb, y_train, y_val


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
    Parameters
    --------
      X_train
          training data
      y_train
          data labels

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C,
                             dual=False, solver='liblinear', verbose=1)
    clf = OneVsRestClassifier(clf, verbose=1)
    clf.fit(X_train, y_train)

    return clf
