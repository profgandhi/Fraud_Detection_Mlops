from pipelines.training_pipeline import train_pipeline

graph_def_dict = {
    'target' : "Is Fraud?",
    "transaction" : ["Use Chip","Merchant City","Errors?","Amount","Hour","Minute"],
    'graph':{
        "card_id" : ['one'],
        "Merchant Name" : ['one']
    }
}

if __name__ == "__main__":
    train_pipeline(data_path="Data\\credit_card_transactions-ibm_v2.csv",graph_def_dict = graph_def_dict)