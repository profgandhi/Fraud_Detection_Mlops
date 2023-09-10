import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataPreprocessStrategy
from typing_extensions import Annotated

@step
def clean_data(df: pd.DataFrame) -> Annotated[pd.DataFrame,"clean_df"]:
    try:
        process_strategy = DataPreprocessStrategy()
        processed_data = process_strategy.handle_data(df)
        logging.info("Data cleaning completed")
        return processed_data
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e


