import os
import logging
import numpy as np
import pandas as pd
import yaml
import pickle
import json
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path:str):
    """Load trained model from file."""
    try:
        model = tf.keras.models.load_model(model_path,compile=False)
        logger.debug("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"an error encountered while loading saved model: {e}")
        raise

def load_data(file_path:str)->tuple:
    """Load the transformed data from files."""
    try:
        data = pd.read_csv(file_path).values
        logger.debug("Transformed data loaded successfully.")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the transformed data files: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the transformed data: {e}")
        raise

def load_scaler(scaler_path:str):
    """Load scaler object from file"""
    try:
        with open(scaler_path,'rb') as f:
            scaler = pickle.load(f)
        logger.debug("Scaler loaded successfully.")
        return scaler
    except FileNotFoundError as e:
        logger.error(f"Scaler file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the scaler: {e}")
        raise

def evaluate_model(model, X_test, y_test,scaler)->dict:
    """Evaluate model performance using various metrics."""
    try:
        # 1. Reshape X_test if it was flattened for CSV (samples, time_steps, features)
        # Assuming 30 time steps and 1 feature
        X_test_reshaped = X_test.reshape(X_test.shape[0], 30, 1)

        # 2. Get Scaled Predictions
        y_pred_scaled = model.predict(X_test_reshaped, verbose=0)

        # 3. Inverse Transform back to original prices
        # We reshape to (-1, 1) because the scaler expects a 2D array
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 4. Calculate Metrics
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)

        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """main function for evaluating the model"""
    try:
        model = load_model("./models/model/lstm_model.h5")
        logger.debug("loaded model into this file")
        X_test = load_data("./data/transformed/X_test.csv")
        y_test = load_data("./data/transformed/y_test.csv")
        logger.debug("loaded test data into this file")
        scaler = load_scaler("./models/scaler/scaler.pkl")
        logger.debug("loaded scaler successfully")
        metrics = evaluate_model(model,X_test,y_test,scaler)
        logger.debug("evaluated the model successfully")
        save_metrics(metrics, "./evaluation/metrics.json")
        logger.debug("saved the metrics successfully")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__=="__main__":
    main()