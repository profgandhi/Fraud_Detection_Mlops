import logging 
from abc import ABC, abstractmethod
import torch

class Evaluation(ABC):
    '''
    Abstract class defining strategy for evaluation our models
    '''

    @abstractmethod
    def calculate_scores(self):
        pass




class HeteroModelEval():

    def calculate_scores():
        model.eval()
        total_examples = total_correct = 0
        for batch in tqdm(eval_loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch[transaction_entity].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)[transaction_entity][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch[transaction_entity].y[:batch_size]).sum())

        return total_correct / total_examples