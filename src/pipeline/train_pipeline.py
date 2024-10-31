import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass
    
    def train(self):
        try:
            logging.info("Started training pipeline.")
            
            # 1. Data Ingestion: Load the Dataset
            logging.info("Loading data...")
            data_ingest_obj = DataIngestion()   
                 
            # Split the dataset into training and testing sets.
            logging.info("Splitting the dataset into training and testing sets")
            train_data, test_data = data_ingest_obj.initiate_data_ingestion()
            
            # 2. Data Transformation and Preprocessing:
            logging.info("Performing data transformation.")
            data_transformation = DataTransformation()
            train_array, test_array, preprocessor_filepath = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)
            
            # 3. Model Training: Train the Model
            logging.info("Performing model training.")
            model_trainer = ModelTrainer()
            print(model_trainer.initiate_model_trainer(train_array, test_array))
            
            logging.info("Train Pipeline Completed.")
            
        except Exception as e:
            raise CustomException(e, sys)

# Usage Example:
if __name__ == "__main__":
    trainer = TrainPipeline()
    trainer.train()