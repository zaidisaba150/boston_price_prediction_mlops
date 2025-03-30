# src/model_evaluation.py
import pandas as pd
import joblib
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to log to both file and console
log_file_path = os.path.join(log_dir, "model_evaluation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)


def evaluate_model(
    model_path="models/house_price_model.pkl",
    test_path="data/feature_engineering/test_fe.csv",
    results_dir="evaluation",
):
    """Evaluate the trained model on the test data and save results"""

    try:
        # Load the trained model
        model = joblib.load(model_path)
        logging.info("Trained model loaded successfully!")
    except FileNotFoundError as e:
        logging.error(f"Error loading model: {e}")
        raise

    try:
        # Load the test data
        test_data = pd.read_csv(test_path)
        logging.info("Feature-engineered test data loaded successfully!")
    except FileNotFoundError as e:
        logging.error(f"Error loading test data: {e}")
        raise

    try:
        # Split features and target
        X_test = test_data.drop(columns=["Price"])
        y_test = test_data["Price"]
        logging.info("Test data split into features and target successfully.")
    except KeyError as e:
        logging.error(f"Error: Column 'Price' not found in test data: {e}")
        raise

    try:
        # Make predictions
        y_pred = model.predict(X_test)
        logging.info("Model predictions completed successfully!")
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise

    try:
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info("Model evaluation metrics calculated successfully!")
        logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logging.info(f"R-squared (R2): {r2:.4f}")

    except Exception as e:
        logging.error(f"Error calculating evaluation metrics: {e}")
        raise

    # Create output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Save evaluation results to a CSV file
        results = pd.DataFrame(
            {
                "Metric": ["MSE", "RMSE", "MAE", "R2"],
                "Value": [mse, rmse, mae, r2],
            }
        )

        results_path = os.path.join(results_dir, "evaluation_results.csv")
        results.to_csv(results_path, index=False)
        logging.info(f"Evaluation results saved to {results_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")
        raise

    return results


# Run the function when executed as a script
if __name__ == "__main__":
    try:
        evaluate_model()
        logging.info("Model evaluation completed successfully!")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
