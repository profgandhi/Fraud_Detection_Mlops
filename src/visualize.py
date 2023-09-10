import logging 
from abc import ABC, abstractmethod
import torch
from torch_geometric.data import HeteroData
from sklearn.manifold import TSNE
class Representation(ABC):
    '''
    Abstract class defining strategy for evaluation our models
    '''

    @abstractmethod
    def get_representation(self):
        pass

class NodeRepresentation(Representation):

    def __init__(self):
        self.embeddings = None

    def get_representation(self,hgraph: HeteroData,model,entity_column: str):
        with torch.no_grad():  
            model.eval()
            _, embeddings = model(hgraph.x_dict, hgraph.edge_index_dict,embeddings_for=entity_column)
        embeddings = embeddings.detach().numpy()
        self.embeddings = embeddings
        return embeddings
    
    def embedding_dim_reduction(self,hgraph,model,entity_column: str):
        '''
        Perform Tsne dimention reduction
        '''  
    
        if(self.embeddings == None):
              self.embeddings = self.get_representation(hgraph,model,entity_column)

        reduced_embeddings = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(self.embeddings) 

        x = reduced_embeddings[:, 0]
        y = reduced_embeddings[:, 1]
        
        return {'x':x, 'y':y}
    


  
