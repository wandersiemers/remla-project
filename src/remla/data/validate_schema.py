from typing import List, Tuple

import pandas as pd
import tensorflow_data_validation as tfdv

DATA_PATH = "assets/data/raw"


def get_anomalies(filename: str, schema_path: str):
    dataset = pd.read_csv(f"{DATA_PATH}/{filename}.tsv", sep="\t")
    schema = tfdv.load_schema_text(schema_path)
    stats = tfdv.generate_statistics_from_dataframe(dataset)
    anomalies = tfdv.validate_statistics(stats, schema)

    return anomalies


data_and_schema_names: List[Tuple[str, str]] = [
    ("train", "train"),
    ("train", "validation"),
    ("test", "test"),
]

for schema_name, data_name in data_and_schema_names:
    current_anomalies = get_anomalies(
        data_name, f"assets/data/{schema_name}_schema.pbtxt"
    )
    
    assert not current_anomalies.anomaly_info
