import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split

class GraphStrategy(ABC):
    '''
    Abstract class for creation of graph
    '''

    @abstractmethod
    def get_graph(self, df: pd.DataFrame, graph_def_dict: dict):
        pass

class HeteroGraphStrategy(GraphStrategy):
    '''
    Strategy to make graph from tabular data
    '''

    def get_graph(self, df: pd.DataFrame, graph_def_dict: dict):

        '''
        Args:
            df: pandas dataframe with data
            graph_def_dict: a dictionay with the graph definition
            eg:
            {
                'target': # Has target column name
                'transaction': ["",""] # list of column to be included in the transaction node
                'graph': {                          # Dictionary having graph structure
                    "node column" : ["",""]         # List of node data columns
                }
            }
        '''
        try:
            target = graph_def_dict['target']
            graph_node_def_dict = graph_def_dict['graph']
            transaction_list = graph_def_dict['transaction']

            # Making heterograph object
            hgraph = HeteroData()
            
            X = df.copy()
            y = X[target]
     

            # Transaction Node
            x = X[transaction_list].reset_index(drop=True)
            x = x.to_numpy()
            hgraph['transaction'].x = torch.from_numpy(x).to(torch.float)

            #Nodes
            for node in graph_node_def_dict: 

                print("x->", node)
                x = pd.DataFrame()

                for node_attr in graph_node_def_dict[node]:
                    if(node_attr == 'one'):
                        x = pd.concat( [ x,pd.Series(np.ones( X[node].nunique() ) )],axis=1 )

                    else: # node_attr from df are int or float
                        if(X[node_attr].dtype == int): #Categorical features
                            modes = X.groupby(node)[node_attr].agg(lambda x: pd.Series.mode(x)[0]) # modes for each node attr , handle multiple modes
                            x = pd.concat([x,modes],axis=1)

                        if(X[node_attr].dtype == float): #numerical features
                            means = X.groupby(node)[node_attr].agg({np.mean})
                            x = pd.concat([x,means],axis=1) 

                #Changing to torch tensor
                x = x.reset_index(drop=True)
                x = x.to_numpy()
                hgraph[node].x = torch.from_numpy(x).to(torch.float)

                

            print(' Making labels and masks ')
            # Adding Labels
            y = y.to_numpy()
            y = torch.from_numpy(y)
            hgraph["transaction"].y = y.to(torch.long)

            # Train Test Split Masks
            train_idx, test_idx = train_test_split(range(len(hgraph["transaction"].y)), stratify=hgraph["transaction"].y, test_size=0.25)
            # train_mask
            train_mask = np.zeros((len(hgraph["transaction"].y),),dtype=bool)
            train_mask[train_idx] = True
            train_mask = torch.from_numpy(train_mask)
            hgraph["transaction"].train_mask = train_mask
            # val_mask
            val_mask = np.zeros((len(hgraph["transaction"].y),),dtype=bool)
            val_mask[test_idx] = True
            val_mask = torch.from_numpy(val_mask)
            hgraph["transaction"].val_mask = val_mask


            #Edges
            for node in graph_node_def_dict:
                print("edges->",node)
                #The categorical codes for each type enity are the index for each entity 
                edges = X[node]
                edges = edges.reset_index()
                edges = edges.values.transpose()
                print(edges)
                edges = torch.from_numpy(edges)
                hgraph["transaction","::"+node,node].edge_index = edges.to(torch.long)

            hgraph = T.ToUndirected()(hgraph)
            print(hgraph)
            return hgraph

        except Exception as e:
            logging.error(f"Error in making graph: {e}")
            raise e



