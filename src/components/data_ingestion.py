import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
# from src.components.model_trainer import ModelTrainer
# from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    """
    DataIngestion variables. 
    Define a configuration class to hold paths for training, testing, and raw data files
    """
    # Define paths for train, test, and raw data files within an 'artifacts' directory
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    # Main data ingestion class that reads data, splits it, and saves train and test datasets
    def __init__(self):
        # Initialize with path fron DataIngestionConfig
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Reading the dataset as Dataframe.")
            
            # Ensure the directory for the train and test files exists 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Saves the raw dataset into the raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=41)
            
            # Save the training set into the train_data_path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save the testing set into the test_data_path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data is completed.")
            
            # Return the paths to the train and test data files
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
        
        
# if __name__ == "__main__":
#     data_ingest_obj = DataIngestion()        
#     data_ingest_obj.initiate_data_ingestion()
    