# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import logging
import warnings

warnings.simplefilter(action="ignore")

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "data_preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def preprocess_data(
    train_path="data/ingestion/train.csv",
    test_path="data/ingestion/test.csv",
    save_dir="data/preprocessing",
):
    """Preprocess train and test data, handle missing values, and save to CSV"""

    try:
        # Load train and test data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Train and test data loaded successfully!")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Drop 'Id' as it's not useful for modeling
    if "Id" in train_data.columns:
        train_data.drop("Id", axis=1, inplace=True)
    if "Id" in test_data.columns:
        test_data.drop("Id", axis=1, inplace=True)
    logging.info("Dropped 'Id' column from train and test data.")

    # Handle missing values
    train_data.fillna(train_data.median(numeric_only=True), inplace=True)
    test_data.fillna(test_data.median(numeric_only=True), inplace=True)

    # Fill missing categorical values with mode
    categorical_cols = ["Location", "Condition", "Garage"]
    for col in categorical_cols:
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)
        test_data[col].fillna(test_data[col].mode()[0], inplace=True)
    logging.info("Missing values handled successfully!")

    # Separate features and target
    X_train = train_data.drop("Price", axis=1)
    y_train = train_data["Price"]

    X_test = test_data.drop("Price", axis=1)
    y_test = test_data["Price"]

    # Identify numerical and categorical features
    numerical_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
    categorical_features = ["Location", "Condition", "Garage"]

    # Define transformers for scaling and encoding
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Apply transformations using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create a pipeline for preprocessing
    preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform the training data
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    # Get transformed feature names
    encoded_columns = preprocessing_pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(
        categorical_features
    )
    feature_columns = numerical_features + list(encoded_columns)

    # Convert transformed arrays to DataFrame
    X_train_processed = pd.DataFrame(X_train_transformed, columns=feature_columns)
    X_test_processed = pd.DataFrame(X_test_transformed, columns=feature_columns)

    # Add target variable back to the data
    train_processed = pd.concat([X_train_processed, y_train.reset_index(drop=True)], axis=1)
    test_processed = pd.concat([X_test_processed, y_test.reset_index(drop=True)], axis=1)

    # Define save paths
    train_processed_path = os.path.join(save_dir, "train_processed.csv")
    test_processed_path = os.path.join(save_dir, "test_processed.csv")

    # Save preprocessed data to CSV
    train_processed.to_csv(train_processed_path, index=False)
    test_processed.to_csv(test_processed_path, index=False)

    logging.info(f"Preprocessed train data saved to {train_processed_path}")
    logging.info(f"Preprocessed test data saved to {test_processed_path}")

    return train_processed, test_processed


# Run the function when executed as a script
if __name__ == "__main__":
    try:
        preprocess_data()
        logging.info("Data preprocessing completed successfully!")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
