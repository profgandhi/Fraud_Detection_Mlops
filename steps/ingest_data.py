import logging
import pandas as pd
from zenml import step

from src.ingest_data import IngestData
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    '''
    Ingesting the data from the data_path

    Args:
        data_path: path tp the data
    Returns:
        pd.DataFrame: the ingested data
    '''
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
