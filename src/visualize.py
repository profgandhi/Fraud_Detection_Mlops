import logging 
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import HeteroData
from sklearn.manifold import TSNE

import plotly.express as px
import pandas as pd

class Representation(ABC):
    '''
    Abstract class defining strategy for evaluation our models
    '''

    @abstractmethod
    def get_representation(self):
        pass

class NodeRepresentation(Representation):

    def __init__(self,hgraph: HeteroData,model,entity_column: str):
        self.entity_column = entity_column
        self.hgraph = hgraph
        self.model = model
        self.embeddings = None
        self.reduc = None

    def get_representation(self):
        with torch.no_grad():  
            self.model.eval()
            _, embeddings = self.model(self.hgraph.x_dict, self.hgraph.edge_index_dict,embeddings_for=self.entity_column)
        embeddings = embeddings.detach().numpy()
        self.embeddings = embeddings
        
    
    def embedding_dim_reduction(self):
        '''
        Perform Tsne dimention reduction
        '''  
    
        if(self.embeddings == None):
            self.get_representation()

        reduced_embeddings = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(self.embeddings) 

        x = reduced_embeddings[:, 0]
        y = reduced_embeddings[:, 1]
        self.reduc = {'x':x, 'y':y}
    
    def get_plot(self):
        if(self.reduc == None):
            self.embedding_dim_reduction()

        q = pd.DataFrame(self.reduc,columns=['x','y'])
        q['cluster'] = self.hgraph['transaction'].y
        fig = px.scatter(q, x='x', y='y',color='cluster',title="Embeddings")

        return fig
    


  
