import pandas as pd

pd.read_csv('assets/data/raw/train.tsv', sep='\t').to_csv('assets/data/processed/train.tsv')
pd.read_csv('assets/data/raw/test.tsv', sep='\t').to_csv('assets/data/processed/test.tsv')
pd.read_csv('assets/data/raw/validation.tsv', sep='\t').to_csv('assets/data/processed/validation.tsv')