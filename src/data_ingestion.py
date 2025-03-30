# src/data_ingestion.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

import os
import logging

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "data_ingestion.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Extract parameters
csv_path = params["data_paths"]["raw_data"]
save_dir = "data/ingestion"
test_size = params["ingestion_params"]["test_size"]
random_state = params["ingestion_params"]["random_state"]


def load_and_split_data():
    """Load dataset from CSV and split into train and test CSVs"""

    try:
        # Load dataset from CSV
        data = pd.read_csv(csv_path)
        logging.info(f"✅ Data loaded successfully from {csv_path}")
        print(f"✅ Data loaded from {csv_path}")

        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"✅ Directory '{save_dir}' created or already exists.")

        # Split the data into train and test sets
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        logging.info(
            f"✅ Data split successfully with test_size={test_size} and random_state={random_state}"
        )
        print(f"✅ Data split successfully!")

        # Save train and test data to CSVs
        train_path = os.path.join(save_dir, "train.csv")
        test_path = os.path.join(save_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logging.info(f"✅ Train data saved to {train_path}")
        logging.info(f"✅ Test data saved to {test_path}")

        print(f"✅ Train data saved to {train_path}")
        print(f"✅ Test data saved to {test_path}")

        return train_data, test_data

    except Exception as e:
        logging.error(f"❌ Error during data ingestion: {e}")
        raise


if __name__ == "__main__":
    load_and_split_data()
