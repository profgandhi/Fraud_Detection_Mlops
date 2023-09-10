import logging
from abc import ABC, abstractmethod

from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero
import torch.nn.functional as F
import torch
from torch_geometric.nn import Linear, SAGEConv, to_hetero, HeteroConv
from torch_geometric.data import HeteroData
from tqdm import tqdm


class Model(ABC):
    '''
    Abstract class for all models
    '''

    @abstractmethod
    def train(self):
        pass



class HeteroGNN2(torch.nn.Module):
    def __init__(self,HIDDEN_SIZE,EMBEDDING_SIZE,NUMBER_OF_CLASSES,DROP_RATE):
        super().__init__()
        self.DROP_RATE = DROP_RATE
        self.conv1 = SAGEConv(in_channels=(-1, -1), out_channels=HIDDEN_SIZE)
        self.conv2 = SAGEConv(in_channels=(-1, -1), out_channels=HIDDEN_SIZE)
        self.conv3 = SAGEConv(in_channels=(-1, -1), out_channels=EMBEDDING_SIZE)
        self.linear = Linear(in_channels=-1, out_channels=NUMBER_OF_CLASSES)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.DROP_RATE, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.DROP_RATE, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        out = self.linear(x)
        return out  

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata,hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict,embeddings_for):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        out = self.lin(x_dict[embeddings_for])
        return out, x_dict[embeddings_for]
        



class HeteroGnnClassificationModel(Model):
    '''
    GNN model for node classification classification
    '''

    def __init__(self,hgraph: HeteroData):
        #Masks
        self.val_input_nodes = ("transaction", hgraph["transaction"].val_mask)
        self.kwargs = {'batch_size': 512, 'num_workers': 2, 'persistent_workers': True}

        #Loaders
        print(hgraph)
        self.train_loader = NeighborLoader(
            hgraph, 
            num_neighbors={key: [30] * 2 for key in hgraph.edge_types}, 
            shuffle=True,
            input_nodes= ("transaction", hgraph["transaction"].train_mask), 
            **self.kwargs
        )
        
        self.val_loader = NeighborLoader(hgraph, num_neighbors={key: [15] * 2 for key in hgraph.edge_types},input_nodes=self.val_input_nodes, **self.kwargs)

        #device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #hgraph
        self.hgraph = hgraph

    def get_model(self):
        try:
            #load model
            
            #DROP_RATE = 0.3
            #HIDDEN_SIZE = 512
            EMBEDDING_SIZE = 256
            NUMBER_OF_CLASSES = 2

            hidden_channels = EMBEDDING_SIZE
            out_channels = NUMBER_OF_CLASSES 
            num_layers = 2
            #model = HeteroGNN2(HIDDEN_SIZE,EMBEDDING_SIZE,NUMBER_OF_CLASSES,DROP_RATE)
            model = HeteroGNN(self.hgraph.metadata(), hidden_channels, out_channels,num_layers) 
         
    
            # Get lazy parameters
            batch = next(iter(self.train_loader))
            batch = batch.to(self.device, 'edge_index')
            model(batch.x_dict, batch.edge_index_dict,"transaction")

            return model
        
        except Exception as e:
            logging.error(f"Error in initializing model: {e}")
            raise e


    def train(self):
        try:
            model = self.get_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005 , weight_decay=0.0001)
            criterion = torch.nn.CrossEntropyLoss()
            EPOCHS = 1
            for epoch in range(EPOCHS+1):
                total_loss = 0
                acc = 0
                ks = 0
                # Train on batches
                for batch in tqdm(self.train_loader):
                    optimizer.zero_grad()
       
                    batch_size = batch["transaction"].batch_size        
                    out,h = model(batch.x_dict, batch.edge_index_dict,embeddings_for="transaction")[:batch_size]
                    loss = F.cross_entropy(out[:batch_size], batch["transaction"].y[:batch_size] )

                    
                    total_loss += loss
                    loss.backward()
                    optimizer.step()

            return model

        except Exception as e:
            logging.error(f"Error while training GNN model: {e}")
            raise e
        

    