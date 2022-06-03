import pandas as pd
import tensorflow_data_validation as tfdv

SCHEMA_PATH = 'assets/data/schema.pbtxt'
DATA_PATH = 'assets/data/raw'


def get_anomalies(filename: str):
    dataset = pd.read_csv(f"{DATA_PATH}/{filename}.tsv", sep='\t')
    schema = tfdv.load_schema_text(SCHEMA_PATH)
    stats = tfdv.generate_statistics_from_dataframe(dataset)
    anomalies = tfdv.validate_statistics(stats, schema)
    return anomalies


def test_train_schema():
    train_anomalies = get_anomalies('train')
    assert not train_anomalies.anomaly_info


def test_validation_schema():
    validation_anomalies = get_anomalies('validation')
    assert not validation_anomalies.anomaly_info


def test_test_schema():
    test_anomalies = get_anomalies('test')
    assert test_anomalies.anomaly_info
