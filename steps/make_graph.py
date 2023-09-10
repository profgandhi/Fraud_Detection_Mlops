import logging
import pandas as pd
from zenml import step

from src.making_graph import HeteroGraphStrategy
from typing_extensions import Annotated
from torch_geometric.data import HeteroData

@step
def make_graph(df: pd.DataFrame, graph_def_dict: dict) -> Annotated[HeteroData,"Hetero-Graph"]:
    try:
        graph_strategy = HeteroGraphStrategy()
        hgraph = graph_strategy.get_graph(df,graph_def_dict)
        logging.info("Graph made successfully")
        return hgraph
    except Exception as e:
        logging.error(f"Error while making graph: {e}")
        raise e