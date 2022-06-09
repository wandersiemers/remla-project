import pandas as pd
import tensorflow_data_validation as tfdv

DATA_PATH = "assets/data/raw"


def get_anomalies(filename: str, schema_path: str):
    dataset = pd.read_csv(f"{DATA_PATH}/{filename}.tsv", sep="\t")
    schema = tfdv.load_schema_text(schema_path)
    stats = tfdv.generate_statistics_from_dataframe(dataset)
    anomalies = tfdv.validate_statistics(stats, schema)
    return anomalies


def test_train_schema():
    SCHEMA_PATH = "assets/data/train_schema.pbtxt"
    train_anomalies = get_anomalies("train", SCHEMA_PATH)
    assert not train_anomalies.anomaly_info


def test_validation_schema():
    SCHEMA_PATH = "assets/data/train_schema.pbtxt"
    validation_anomalies = get_anomalies("validation", SCHEMA_PATH)
    assert not validation_anomalies.anomaly_info


def test_test_schema():
    SCHEMA_PATH = "assets/data/test_schema.pbtxt"
    test_anomalies = get_anomalies("test", SCHEMA_PATH)
    assert not test_anomalies.anomaly_info
