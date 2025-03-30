# src/model_development.py
import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to log to both file and console
log_file_path = os.path.join(log_dir, "model_development.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)


def train_model(
    train_path="data/feature_engineering/train_fe.csv",
    test_path="data/feature_engineering/test_fe.csv",
    model_dir="models",
):
    """Train a machine learning model on the feature-engineered data."""

    try:
        # Load feature-engineered train & test data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Feature-engineered train and test data loaded successfully!")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Split features and target variable
    try:
        X_train = train_data.drop(columns=["Price"])
        y_train = train_data["Price"]

        X_test = test_data.drop(columns=["Price"])
        y_test = test_data["Price"]
        logging.info("Split data into features and target successfully.")
    except KeyError as e:
        logging.error(f"Missing target column 'Price' in data: {e}")
        raise

    # Train a Random Forest Regressor
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully!")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

    # Define model save path
    model_path = os.path.join(model_dir, "house_price_model.pkl")

    try:
        # Save trained model
        joblib.dump(model, model_path)
        logging.info(f"Trained model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving trained model: {e}")
        raise

    return model


# Run the function when executed as a script
if __name__ == "__main__":
    try:
        train_model()
        logging.info("Model development completed successfully!")
    except Exception as e:
        logging.error(f"Error during model development: {e}")
