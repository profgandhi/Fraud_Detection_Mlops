import logging
from abc import ABC, abstractmethod
import pandas as pd

class IngestData:

    '''
    Ingesting data from the data_path
    
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path,nrows=20000)