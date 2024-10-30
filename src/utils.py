import os
import sys
import pickle

import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# File contains functions used in this project

def save_object(file_path, obj):
    """Save pyhton object to a file.

    Args:
        file_path (str): Path to file.
        obj (_type_): Obj to be saved
    """
    try:
        # Get the directory path from the file_path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory(folder) if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file in a write-binary mode to save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj) # save object to specified file path
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train,X_test,y_test,models,param):
    """Function to evaluate multiple models with hyperparameter tuning and return test scores"""
    try:
        # Dictionary to store model name and their test score
        report = {}
        
        for i in range(len(list(models))):
            # Get the model object by its position in dictionary
            model = list(models.values())[i]
            # Get the corresponding hyperparameter grid from model
            para = param[list(models.keys())[i]]
            
            # Perform hyperparamter tuning using GridSearch CV
            gs = GridSearchCV(model, para, cv=3) # 3-fold cross-validation
            gs.fit(X_train, y_train)  # Fit model on training data with different parameters
            
            # Set the model's parameters to the best found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # Retrain the model with the best parameters
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R² score on training data to assess model fit and on testing data to assess model generalization
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Add the model's test score to the report dictionary, using the model name as the key
            report[list(models.keys())[i]] = test_model_score
            
        # Return the report dictionary containing each model and its test score            
        return report
            
    except Exception as e:
        raise CustomException(e, sys)