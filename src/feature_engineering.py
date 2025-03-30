# src/feature_engineering.py
import pandas as pd
import numpy as np
import os
import logging


# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to log to both file and console
log_file_path = os.path.join(log_dir, "feature_engineering.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)


def feature_engineering(
    train_path="data/preprocessing/train_processed.csv",
    test_path="data/preprocessing/test_processed.csv",
    save_dir="data/feature_engineering",
):
    """Perform feature engineering on the preprocessed data and save to CSV"""

    try:
        # Load preprocessed train and test data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Preprocessed train and test data loaded successfully!")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create 'House_Age' feature (current year - YearBuilt)
    train_data["House_Age"] = 2025 - train_data["YearBuilt"]
    test_data["House_Age"] = 2025 - test_data["YearBuilt"]
    logging.info("Feature 'House_Age' created successfully!")

    # Drop 'YearBuilt' since it's now replaced by 'House_Age'
    if "YearBuilt" in train_data.columns:
        train_data.drop("YearBuilt", axis=1, inplace=True)
    if "YearBuilt" in test_data.columns:
        test_data.drop("YearBuilt", axis=1, inplace=True)
    logging.info("Dropped 'YearBuilt' after creating 'House_Age'.")

    # Create 'Total_Bathrooms' by combining Bathrooms and Floors
    train_data["Total_Bathrooms"] = train_data["Bathrooms"] * train_data["Floors"]
    test_data["Total_Bathrooms"] = test_data["Bathrooms"] * test_data["Floors"]
    logging.info("Feature 'Total_Bathrooms' created successfully!")

    # Create 'Is_Luxury_House' based on Area > 3000 and Condition == Excellent
    if "Condition_Excellent" in train_data.columns and "Condition_Excellent" in test_data.columns:
        train_data["Is_Luxury_House"] = np.where(
            (train_data["Area"] > 3000) & (train_data["Condition_Excellent"] == 1), 1, 0
        )
        test_data["Is_Luxury_House"] = np.where(
            (test_data["Area"] > 3000) & (test_data["Condition_Excellent"] == 1), 1, 0
        )
        logging.info("Feature 'Is_Luxury_House' created successfully!")

    # Log transformation for 'Area' to reduce skewness (optional)
    train_data["Log_Area"] = np.log1p(train_data["Area"])
    test_data["Log_Area"] = np.log1p(test_data["Area"])
    logging.info("Feature 'Log_Area' created successfully using log transformation.")

    # Drop original 'Area' after transformation
    if "Area" in train_data.columns:
        train_data.drop("Area", axis=1, inplace=True)
    if "Area" in test_data.columns:
        test_data.drop("Area", axis=1, inplace=True)
    logging.info("Dropped original 'Area' after applying log transformation.")

    # Define save paths
    train_fe_path = os.path.join(save_dir, "train_fe.csv")
    test_fe_path = os.path.join(save_dir, "test_fe.csv")

    # Save feature-engineered data to CSV
    train_data.to_csv(train_fe_path, index=False)
    test_data.to_csv(test_fe_path, index=False)

    logging.info(f"Feature-engineered train data saved to {train_fe_path}")
    logging.info(f"Feature-engineered test data saved to {test_fe_path}")

    return train_data, test_data


# Run the function when executed as a script
if __name__ == "__main__":
    try:
        feature_engineering()
        logging.info("Feature engineering completed successfully!")
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
