from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.make_graph import make_graph
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str,graph_def_dict: dict):
    df = ingest_data(data_path)
    df = clean_data(df)
    graph = make_graph(df,graph_def_dict)
    trained_model = train_model(graph)

