import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from abc import ABC

class CONFIG(ABC):
    '''
    Abstract class for hyperparameters for model, input, training
    '''
    pass

class GnnConfig(CONFIG):

    def __init__(self):
        #Training parameters
        self.batch_size = 512
        self.lr=0.0005  
        self.weight_decay=0.0001
        self.EPOCHS = 2

        #Model parameters
        self.EMBEDDING_SIZE = 256
        self.NUMBER_OF_CLASSES = 2
        self.num_hetero_conv = 2
        self.GraphDict = {
            'target' : "Is Fraud?",
            "transaction" : ["Use Chip","Merchant City","Errors?","Amount","Hour","Minute"],
            'graph':{
                "card_id" : ['one'],
                "Merchant Name" : ['one']
            }
        }
        






input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, -1), name="x"), 
                       TensorSpec(np.dtype(np.int32), (-1, -1), name="edge_index"), 
                       TensorSpec(np.dtype(np.int32), (-1, -1), name="batch_index")])

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)