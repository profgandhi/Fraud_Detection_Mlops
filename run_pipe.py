from src.ingest_data import IngestData
from src.data_cleaning import DataPreprocessStrategy
from src.making_graph import HeteroGraphStrategy
from src.model_dev import HeteroGnnClassificationModel
from src.visualize import NodeRepresentation


import mlflow
import mlflow.pytorch

from config import GnnConfig

graph_def_dict = {
    'target' : "Is Fraud?",
    "transaction" : ["Use Chip","Merchant City","Errors?","Amount","Hour","Minute"],
    'graph':{
        "card_id" : ['one'],
        "Merchant Name" : ['one']
    }
}
data_path="Data\\credit_card_transactions-ibm_v2.csv"


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    ingest_data = IngestData(data_path)
    df = ingest_data.get_data()

    process_strategy = DataPreprocessStrategy()
    df = process_strategy.handle_data(df)

    graph_strategy = HeteroGraphStrategy()
    hgraph = graph_strategy.get_graph(df,graph_def_dict)

    trained_model,train_loss = HeteroGnnClassificationModel(hgraph).train()
    print(trained_model)

    plot = NodeRepresentation(hgraph,trained_model,"transaction").get_plot()
    
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            trained_model, "model"
        ) 

        mlflow.log_param("learning_rate", GnnConfig().lr)
        mlflow.log_param("batch_size", GnnConfig().batch_size)
        mlflow.log_param("weight_decay", GnnConfig().weight_decay )
        mlflow.log_param("EPOCHS", GnnConfig().EPOCHS)
        mlflow.log_param("EMBEDDING_SIZE", GnnConfig().EMBEDDING_SIZE)
        mlflow.log_param("num_hetero_conv", GnnConfig().num_hetero_conv)
        mlflow.log_param("Graph_Defination", str(GnnConfig().GraphDict))
        
        mlflow.log_figure(plot ,'Embedding_Plot.html')
    
        