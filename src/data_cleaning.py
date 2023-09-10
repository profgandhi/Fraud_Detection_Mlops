import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handeling data
    '''

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass



class DataPreprocessStrategy(DataStrategy):
    '''
    Strategy for preprocessing data
    '''

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        PreProcess Data
        '''
        try:

            data["card_id"] = data["User"].astype(str) + "_" + data["Card"].astype(str)
            data["Amount"] = data["Amount"].str.replace("$","").astype(float)
            data["Hour"] = data["Time"].str[0:2].astype(int)
            data["Minute"] = data["Time"].str [3:5].astype(int)

            data = data.drop(
                [
                    "Merchant State",
                    "Zip",
                    "Time",
                    "User",
                    "Card"
                ],
                axis=1
            )

            data["Is Fraud?"] = data["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)

            data["Errors?"].fillna("No error",inplace=True)

            data["Merchant Name"]=LabelEncoder().fit_transform(data["Merchant Name"])
            data["Merchant City"]=LabelEncoder().fit_transform(data["Merchant City"])
            data["Use Chip"]=LabelEncoder().fit_transform(data["Use Chip"])
            data["Errors?"]=LabelEncoder().fit_transform(data["Errors?"])
            data["card_id"] = LabelEncoder().fit_transform(data["card_id"])

            # Binning strategy for features


            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
        





