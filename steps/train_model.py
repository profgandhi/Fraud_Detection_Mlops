import logging
import pandas as pd
from zenml import step

from torch_geometric.data import HeteroData
from src.model_dev import HeteroGnnClassificationModel

from torch import nn

@step
def train_model(hgraph: HeteroData) -> nn.Module :
    try:
        model = None

        trained_model = HeteroGnnClassificationModel(hgraph).train()
        print(type(trained_model))

        return trained_model
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e