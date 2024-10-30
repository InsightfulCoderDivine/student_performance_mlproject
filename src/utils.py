import os
import sys
import pickle

import numpy as np
import pandas as pd
from src.exception import CustomException

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