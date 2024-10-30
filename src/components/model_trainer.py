import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """Train model.

        Args:
            train_array (np.array): Train data.
            test_array (np.array): Test preprocessed data.
        """
        try:
            logging.info("Splitting train test data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # X_train: All rows, all columns except the last
                train_array[:, -1],   # y_train: All rows, only the last column
                test_array[:, :-1],   # X_test: All rows, all columns except the last
                test_array[:, -1]     # y_test: All rows, only the last column
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            # Dictionary of hyperparameters for each model
            params = {
                "Linear Regression": {
                    # Linear Regression doesn't have many tunable hyperparameters, so leaving this empty
                },
                
                "Ridge": {
                    'alpha': [0.01, 0.1, 1, 10, 100],  # Regularization strength
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']  # Solver algorithm
                },
                
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to use
                    'weights': ['uniform', 'distance'],  # Weight function used in prediction
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used for computing nearest neighbors
                    'p': [1, 2]  # Power parameter for the Minkowski distance metric (1 for Manhattan, 2 for Euclidean)
                },
                
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # Function to measure split quality
                    # 'splitter': ['best', 'random'],  # Split strategy
                    # 'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
                    # 'min_samples_split': [2, 5, 10],  # Minimum number of samples to split an internal node
                    # 'min_samples_leaf': [1, 2, 4]  # Minimum number of samples at a leaf node
                },
                
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
                    # 'criterion': ['squared_error', 'absolute_error'],  # Function to measure split quality
                    # 'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of each tree
                    # 'min_samples_split': [2, 5, 10],  # Minimum number of samples to split an internal node
                    # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples at a leaf node
                    # 'bootstrap': [True, False]  # Whether to use bootstrapping (sampling with replacement)
                },
                
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],  # Step size at each boosting step
                    'n_estimators': [50, 100, 200, 300],  # Number of boosting rounds
                    # 'max_depth': [3, 5, 7, 10],  # Maximum depth of each tree
                    # 'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used per boosting round
                    # 'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
                },
                
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],  # Depth of each tree
                    'learning_rate': [0.01, 0.05, 0.1],  # Step size at each boosting step
                    'iterations': [100, 200, 300],  # Number of boosting rounds
                },
                
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200, 300],  # Number of boosting rounds
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],  # Step size at each boosting step
                    # 'loss': ['linear', 'square', 'exponential']  # Loss function to use when updating weights
                }
            }

            logging.info("Model evaluation started.")
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            
            # Get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model score greater than 0.6 found.")
            logging.info(f"Best model found on both train and test dataset: {best_model_name} with r2 score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Saved best model Done.")
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)