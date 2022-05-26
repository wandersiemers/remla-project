import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from ast import literal_eval

nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = " ".join([word for word in text.split() if not word in STOPWORDS])
    return text


def write_to_file(X, y=None, filename='test.tsv'):
    assert y is None or len(X) == len(y), "Number of feature rows should equal number of label lists"
    dataframe = pd.DataFrame({'title': X}) if y is None else pd.DataFrame({'title': X, 'tags': y})
    dataframe.to_csv(f'assets/data/processed/{filename}', sep='\t', index=False)


def main():
    train = read_data('assets/data/raw/train.tsv')
    validation = read_data('assets/data/raw/validation.tsv')
    test = pd.read_csv('assets/data/raw/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    write_to_file(X_train, y_train, 'train.tsv')
    write_to_file(X_val, y_val, 'validation.tsv')
    write_to_file(X_test)


if __name__ == '__main__':
    main()
