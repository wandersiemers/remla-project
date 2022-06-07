from typing import Dict, Any, T_co

import torch
import tqdm
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          Trainer,
                          TrainingArguments)
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from remla.models.base_model import BaseModel
from remla.utils import get_corpus_counts


class StackOverflowDataset(Dataset):
    def __init__(self, X_train, y_train, tokenizer: DistilBertTokenizer):
        self.X_train = X_train
        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(self.X_train.tolist(),
                                        truncation=True, padding=True, max_length=64,
                                        return_tensors='pt')
        self.labels = torch.Tensor(y_train)

    def __getitem__(self, index) -> T_co:
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

    def to_device(self, device):
        for item in self.encodings:
            self.encodings[item] = self.encodings[item].to(device)
        self.labels = self.labels.to(device)


class BertBasedModel(BaseModel):
    def __init__(self, logging: bool, config: Dict[str, Any]):
        BaseModel.__init__(self, logging)

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification \
            .from_pretrained('distilbert-base-uncased', num_labels=100)
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def get_features(self, X: list[str]):
        return self.tokenizer(list(X), truncation=True, padding=True, max_length=64,
                              return_tensors='pt')

    def get_labels(self, y: list[list[str]]):
        return self._mlb.fit_transform(y)

    def train(self, X_train: list[str], y_train: list[list[str]]):
        _, tags_counts = get_corpus_counts(X_train, y_train)
        classes = tags_counts.keys()
        self._mlb = MultiLabelBinarizer(classes=sorted(classes))

        dataset = StackOverflowDataset(X_train[:1000],
                                       self.get_labels(y_train)[:1000],
                                       self.tokenizer)
        dataset.to_device(self.device)
        training_args = TrainingArguments(output_dir="test_trainer",
                                          logging_strategy="steps",
                                          logging_steps=10,
                                          do_eval=False,
                                          report_to="wandb")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

    def predict(self, X_test: list[str]):
        y_pred = []
        dataset = StackOverflowDataset(X_train=X_test,
                                       y_train=[0 for _ in X_test],
                                       tokenizer=self.tokenizer)
        dataset.to_device(self.device)

        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for batch in tqdm.tqdm(loader):
            outputs = self.model(batch['input_ids'])
            y_pred.extend((outputs.logits > 0).float().cpu().numpy())
        return y_pred
