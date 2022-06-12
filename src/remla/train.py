import argparse
import importlib
import os
from typing import Type

import wandb

from remla.config import wandb_entity, wandb_project_name
from remla.data.pre_processing import read_files
from remla.models.base_model import BaseModel


def main():
    parser = argparse.ArgumentParser(
        description="Train a machine learning model on processed data from the data pipeline"  # noqa
    )
    parser.add_argument("-mdl", "--model-name", help="Name of the model", required=True)
    parser.add_argument(
        "-mod", "--module-name", help="Name of the module", required=True
    )
    args = parser.parse_args()

    model_name: str = args.model_name
    module_name: str = args.module_name

    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        tags=["training"],
        config={"model": model_name},
    )

    X_train, y_train, _, _, _ = read_files("processed")

    module = importlib.import_module(module_name)

    Model: Type[BaseModel] = getattr(module, model_name)

    # Make sure that we can save the models
    os.makedirs(os.path.join("assets", "models"), exist_ok=True)

    model = Model(True, {})

    model.train(X_train, y_train)
    model.save(f"assets/models/{model_name}.joblib")


if __name__ == "__main__":
    main()
