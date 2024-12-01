import os

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge

# Constants
DATA_FILE_BASE_PATH = "data/"
TRAIN_FILE_PATH = DATA_FILE_BASE_PATH + "train_data.csv"
TEST_FILE_PATH = DATA_FILE_BASE_PATH + "test_data.csv"
SUBMISSION_EXTENSION = ".csv"

BASELINE_SUBMISSION_FILE_PATH = DATA_FILE_BASE_PATH + "baseline-submission-01.csv"
C_MSE_BASELINE_SUBMISSION_FILE_PATH = DATA_FILE_BASE_PATH + "cMSE-baseline-submission-02.csv"

FEATURE_COLUMN_NAMES = [
    "Age", "Gender", "Stage", "GeneticRisk", "TreatmentType",
    "ComorbidityIndex", "TreatmentResponse"
]

TARGET_COLUMNS = ["SurvivalTime", "Censored"]
ID_COLUMN_NAME = "id"
PREDICTION_COLUMN_NAME = "SurvivalTime"


def get_next_submission_filename(base_path, prefix, extension):
    """
    Generate the next submission filename by incrementing the numerical suffix.

    Parameters:
    base_path (str): The directory where submission files are stored.
    prefix (str): The prefix of the submission file (e.g., 'submission').
    extension (str): The file extension (e.g., '.csv').

    Returns:
    str: The next submission filename with an incremented number.
    """
    if not os.path.isdir(f"{base_path}/{prefix}"):
        os.mkdir(f"{base_path}/{prefix}")

    # Initialize the maximum number found
    max_number = 0

    # Scan the directory for files
    with os.scandir(f"{base_path}/{prefix}") as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith(prefix) and entry.name.endswith(extension):
                # Extract the numerical suffix
                try:
                    suffix = entry.name[len(prefix) + 1: -len(extension)]
                    number = int(suffix)
                    max_number = max(max_number, number)
                except (ValueError, IndexError):
                    # Ignore files that don't match the expected pattern
                    continue

    # Compute the next number and generate the filename
    next_number = max_number + 1
    return f"{base_path}/{prefix}/{prefix}-{next_number:02d}{extension}"


def load_and_preprocess(filename):
    """
    Load and preprocess the dataset from a CSV file.

    Parameters:
    filename: The path to the CSV file.

    Returns:
    The loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(filename)
    if df.columns[0].lower() != ID_COLUMN_NAME:
        df = df.rename(columns={df.columns[0]: ID_COLUMN_NAME})
    print(f"Loaded data from {filename} with shape: {df.shape}")
    return df


def visualize_missing_data(df, title_prefix=""):
    """
    Visualize missing data in the DataFrame using various plots.

    Parameters:
    df: The DataFrame to visualize.
    title_prefix: A prefix for plot titles to differentiate strategies.
    """
    if df.empty or df.shape[1] == 0:
        print(f"{title_prefix} DataFrame is empty or has no columns, skipping visualization.")
        return
    plt.figure(figsize=(10, 6))
    msno.bar(df)
    plt.title(f"{title_prefix} Missing Values Bar Plot")
    plt.show()

    if df.isnull().sum().sum() > 0:  # Only show additional plots if there are missing values
        plt.figure(figsize=(10, 6))
        msno.matrix(df)
        plt.title(f"{title_prefix} Missing Values Matrix")
        plt.show()

        plt.figure(figsize=(10, 6))
        msno.heatmap(df)
        plt.title(f"{title_prefix} Missing Values Heatmap")
        plt.show()

        if df.shape[0] > 1:  # Dendrogram requires at least 2 rows
            plt.figure(figsize=(10, 6))
            msno.dendrogram(df)
            plt.title(f"{title_prefix} Missing Values Dendrogram")
            plt.show()


def visualize_correlation(df, title="Correlation Heatmap"):
    """
    Visualize the correlation heatmap of the DataFrame.

    Parameters:
    df: The DataFrame to visualize.
    title: The title for the visualization.
    """
    if df.empty or df.shape[1] == 0:
        print("DataFrame is empty or has no columns, skipping correlation visualization.")
        return

    correlation_matrix = df.corr()  # Compute correlations for all variables
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()


def train_validate_split(df, train_size=0.6, validate_size=0.2, test_size=0.2, seed=42):
    """
    Split the DataFrame into training, validation, and test sets.

    Parameters:
    df: The DataFrame to split.
    train_size: Proportion of the dataset to include in the train split.
    validate_size: Proportion of the dataset to include in the validation split.
    test_size: Proportion of the dataset to include in the test split.
    seed: Random seed for reproducibility.

    Returns:
    Training, validation, and test DataFrames.
    """
    assert train_size + validate_size + test_size == 1.0, "Train, validate, and test sizes must sum to 1."

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end = int(train_size * len(df_shuffled))
    validate_end = train_end + int(validate_size * len(df_shuffled))

    df_train = df_shuffled.iloc[:train_end]
    df_validate = df_shuffled.iloc[train_end:validate_end]
    df_test = df_shuffled.iloc[validate_end:]

    print(f"Data split into Training: {df_train.shape}, Validation: {df_validate.shape}, Test: {df_test.shape}")
    return df_train, df_validate, df_test


def error_metric(y, y_hat, c):
    """
    Calculate the censored Mean Squared Error (cMSE).
    Given by Professor.

    Parameters:
    y: The true survival time.
    y_hat: The predicted survival time.
    c: The censored variable.

    Returns:
    The cMSE value.
    """
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]


def plot_y_yhat(y_val, y_val_pred, title):
    """
    Plot the actual vs. predicted values for the dataset.

    Parameters:
    y_val: The actual values.
    y_val_pred: The predicted values.
    title: The title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, color='blue', alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual Survival Time')
    plt.ylabel('Predicted Survival Time')
    plt.title(title)
    plt.show()


def prepare_submission(test_df, predictions, filename):
    """
    Prepare the submission file for Kaggle.

    Parameters:
    test_df: The test DataFrame containing 'id'.
    predictions: The predicted survival times.
    filename: The filename for the submission CSV.
    """
    submission_df = pd.DataFrame({
        ID_COLUMN_NAME: test_df[ID_COLUMN_NAME],
        '0': predictions
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def clean_dataframe(df, feature_columns, drop_censored=False):
    """
    Clean the DataFrame by dropping rows with missing SurvivalTime or censored data points.

    Parameters:
    df: The DataFrame to clean.
    feature_columns: List of feature column names.
    drop_censored: If True, drop censored data points.

    Returns:
    The cleaned DataFrame and the list of existing features.
    """
    # Drop rows with missing SurvivalTime or Censored indicators
    df_clean = df.dropna(subset=TARGET_COLUMNS)
    print(f"DataFrame shape after dropping rows with missing SurvivalTime or Censored: {df_clean.shape}")
    if drop_censored:
        df_clean = df_clean[df_clean["Censored"] == 0]  # only keep uncensored data points
        print(f"DataFrame shape after dropping censored data points: {df_clean.shape}")

    existing_features = [col for col in feature_columns if col in df_clean.columns]
    return df_clean, existing_features


def strategy_drop_rows(df, drop_censored=False):
    """
    Strategy 1: Drop rows with any missing data and retain censored data points.

    Parameters:
    df: The original DataFrame.
    drop_censored: If True, drop censored data points.

    Returns:
    The cleaned DataFrame after dropping rows and the list of features used.
    """
    print("\n--- Strategy 1: Drop Rows with Missing Data ---")

    # Visualize missing data before cleaning
    visualize_missing_data(df, title_prefix="Strategy 1")

    # Drop rows with any missing values in feature columns
    df_clean = df.dropna(subset=FEATURE_COLUMN_NAMES)
    print(f"DataFrame shape after dropping rows with missing feature values: {df_clean.shape}")

    # Clean the DataFrame by keeping all uncensored data and dropping censored data points based on its flag
    df_clean, existing_features = clean_dataframe(df_clean, FEATURE_COLUMN_NAMES, drop_censored=drop_censored)
    print(f"Features after dropping rows: {existing_features}")

    return df_clean, existing_features


def strategy_impute_missing(df, drop_censored=False):
    """
    Strategy 2: Impute missing values with the mean and retain censored data points.

    Parameters:
    df: The original DataFrame.
    drop_censored: If True, drop censored data points.

    Returns:
    The cleaned DataFrame after imputation and the list of features used.
    """
    print("\n--- Strategy 2: Impute Missing Values ---")

    # Visualize missing data before cleaning
    visualize_missing_data(df, title_prefix="Strategy 2")

    # Impute missing values with the mean for feature columns
    df_imputed = df.copy()
    for col in FEATURE_COLUMN_NAMES:
        if col in df_imputed.columns:
            mean_value = df_imputed[col].mean()
            df_imputed[col] = df_imputed[col].fillna(mean_value)
            print(f"Imputed missing values in '{col}' with mean value {mean_value:.2f}")

    # Clean the DataFrame by keeping all censored and uncensored data
    df_imputed, existing_features = clean_dataframe(df_imputed, FEATURE_COLUMN_NAMES, drop_censored=drop_censored)
    print(f"Features after imputation: {existing_features}")

    return df_imputed, existing_features


def gradient_descent_c_mse(x, y, c, learning_rate=0.001, epochs=1000, tolerance=1e-6):
    """
    Perform Gradient Descent to minimize the cMSE loss.

    Parameters:
    x: Feature matrix.
    y: Target vector.
    c: Censoring indicators.
    learning_rate: Learning rate for updates.
    epochs: Number of iterations.
    tolerance: Threshold for convergence.

    Returns:
    Learned weights, bias, and history of loss values.
    """
    n_samples, n_features = x.shape

    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0
    history = []

    for epoch in range(epochs):
        # Compute predictions
        y_pred = np.dot(x, w) + b

        # Compute errors
        errors = y_pred - y  # Shape: (n_samples,)

        # Compute cMSE loss
        mse = ((1 - c) * errors ** 2 + c * np.maximum(0, y - y_pred) ** 2).mean()
        history.append(mse)

        # Check for convergence
        if epoch > 0 and abs(history[-2] - history[-1]) < tolerance:
            print(f"Converged at epoch {epoch}")
            break

        # Compute gradients
        mask = (y - y_pred) > 0  # Only consider if y > y_pred for censored data

        # Gradient for weights
        grad_w = (2 * ((1 - c) * errors - 2 * c * (y - y_pred) * mask * (-1))[:, np.newaxis] * x).mean(axis=0)

        # Gradient for bias
        grad_b = (2 * ((1 - c) * errors - 2 * c * (y - y_pred) * mask)).mean()

        # Update weights and bias
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # Optionally, print loss every certain epochs
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: cMSE = {mse:.4f}")

    return w, b, history


def train_and_evaluate_model(df_train, df_validate, df_test, features, strategy_name, model, model_type):
    """
    Generalized function to train and evaluate a machine learning model.

    Parameters:
    df_train: Training DataFrame.
    df_validate: Validation DataFrame.
    df_test: Test DataFrame.
    features: List of feature column names.
    strategy_name: Name of the preprocessing strategy.
    model: The machine learning model instance.
    model_type: Description of the model (e.g., "Linear Regression").

    Returns:
    A dictionary containing the strategy name, validation cMSE, and test cMSE.
    """
    # Train the model
    model.fit(df_train[features], df_train[PREDICTION_COLUMN_NAME])
    print(f"{model_type} model trained successfully.")

    # Validate the model
    y_val_pred = model.predict(df_validate[features])
    c_val = df_validate["Censored"].values
    y_val = df_validate[PREDICTION_COLUMN_NAME].values
    c_mse_val = error_metric(y_val, y_val_pred, c_val)
    print(f"{strategy_name} - {model_type} - Validation cMSE: {c_mse_val:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_val), y_val_pred, f'{strategy_name} - {model_type} - Validation Predicted vs Actual')

    # Test the model
    y_test_pred = model.predict(df_test[features])
    c_test = df_test["Censored"].values
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_mse_test = error_metric(y_test, y_test_pred, c_test)
    print(f"{strategy_name} - {model_type} - Test cMSE: {c_mse_test:.4f}")

    # Plot predicted vs. actual for the test set
    plot_y_yhat(pd.Series(y_test), y_test_pred, f'{strategy_name} - {model_type} - Test Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - {model_type}",
        'validation_cMSE': c_mse_val,
        'test_cMSE': c_mse_test
    }


def train_evaluate_custom_gd(df, features, strategy_name, learning_rate=0.001, epochs=1000, tolerance=1e-6):
    """
    Train and evaluate the model using custom Gradient Descent with cMSE.

    Parameters:
    df: The cleaned DataFrame.
    features: List of feature column names.
    strategy_name: Name of the strategy for logging.
    learning_rate: Learning rate for Gradient Descent.
    epochs: Number of iterations.
    tolerance: Threshold for convergence.

    Returns:
    Dictionary containing validation and test cMSE.
    """
    print(f"\n--- Training and Evaluating Gradient Descent for {strategy_name} ---")

    # Split the data into training, validation, and test sets
    df_train, df_validate, df_test = train_validate_split(df, train_size=0.6, validate_size=0.2, test_size=0.2,
                                                          seed=42)
    print(f"{strategy_name} - Training set: {len(df_train)} samples")
    print(f"{strategy_name} - Validation set: {len(df_validate)} samples")
    print(f"{strategy_name} - Test set: {len(df_test)} samples")

    # Define the feature matrix and target vector for training
    x_train = df_train[features].values
    y_train = df_train[PREDICTION_COLUMN_NAME].values
    c_train = df_train["Censored"].values

    # Perform Gradient Descent
    w, b, history = gradient_descent_c_mse(x_train, y_train, c_train, learning_rate, epochs, tolerance)

    # Compute predictions for the validation set
    x_val = df_validate[features].values
    y_val = df_validate[PREDICTION_COLUMN_NAME].values
    c_val = df_validate["Censored"].values
    y_val_pred = np.dot(x_val, w) + b

    # Compute cMSE for the validation set
    c_mse_val = error_metric(y_val, y_val_pred, c_val)
    print(f"{strategy_name} - Gradient Descent - Validation cMSE: {c_mse_val:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_val), y_val_pred, f'{strategy_name} - Gradient Descent - Validation Predicted vs Actual')

    # Compute predictions for the test set
    x_test = df_test[features].values
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_test = df_test["Censored"].values
    y_test_pred = np.dot(x_test, w) + b

    # Compute cMSE for the test set
    c_mse_test = error_metric(y_test, y_test_pred, c_test)
    print(f"{strategy_name} - Gradient Descent - Test cMSE: {c_mse_test:.4f}")

    # Plot predicted vs. actual for the test set
    plot_y_yhat(pd.Series(y_test), y_test_pred, f'{strategy_name} - Gradient Descent - Test Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - Gradient Descent",
        'validation_cMSE': c_mse_val,
        'test_cMSE': c_mse_test
    }


def train_evaluate(df, features, regressor_name, strategy_name, regressor, alpha):
    """
    Train and evaluate the model using Lasso or Ridge Regression.

    Parameters:
    df: The cleaned DataFrame.
    features: List of feature column names.
    regressor_name: The name of the model (Lasso or Ridge).
    strategy_name: Name of the strategy for logging.
    regressor: The regression model (Lasso or Ridge).
    alpha: Regularization strength.

    Returns:
    Dictionary containing validation and test cMSE.
    """
    print(f"\n--- Training and Evaluating {regressor_name} Regression for {strategy_name} ---")

    # Split the data into training, validation, and test sets
    df_train, df_validate, df_test = train_validate_split(df, train_size=0.6, validate_size=0.2, test_size=0.2,
                                                          seed=42)
    print(f"{strategy_name} - Training set: {len(df_train)} samples")
    print(f"{strategy_name} - Validation set: {len(df_validate)} samples")
    print(f"{strategy_name} - Test set: {len(df_test)} samples")

    # Define the feature matrix and target vector for training
    x_train = df_train[features]
    y_train = df_train[PREDICTION_COLUMN_NAME]

    # Train the model
    model = regressor(alpha=alpha)
    model.fit(x_train, y_train)
    print(f"{regressor_name} Regression model trained with alpha={alpha}.")

    # Validate the model
    y_val_pred = model.predict(df_validate[features])
    c_val = df_validate["Censored"].values
    y_val = df_validate[PREDICTION_COLUMN_NAME].values
    c_mse_val = error_metric(y_val, y_val_pred, c_val)
    print(f"{strategy_name} - {regressor_name} Regression - Validation cMSE: {c_mse_val:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_val), y_val_pred,
                f'{strategy_name} - {regressor_name} Regression - Validation Predicted vs Actual')

    # Evaluate on the test set
    y_test_pred = model.predict(df_test[features])
    c_test = df_test["Censored"].values
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_mse_test = error_metric(y_test, y_test_pred, c_test)
    print(f"{strategy_name} - {regressor_name} Regression - Test cMSE: {c_mse_test:.4f}")

    # Plot predicted vs. actual for the test set
    plot_y_yhat(pd.Series(y_test), y_test_pred,
                f'{strategy_name} - {regressor_name} Regression - Test Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - {regressor_name}",
        'validation_cMSE': c_mse_val,
        'test_cMSE': c_mse_test
    }


def train_evaluate_lasso(df, features, strategy_name, alpha=0.1):
    """
    Train and evaluate the model using Lasso Regression.

    Parameters:
    df: The cleaned DataFrame.
    features: List of feature column names.
    strategy_name: Name of the strategy for logging.
    alpha: Regularization strength.

    Returns:
    Dictionary containing validation and test cMSE.
    """
    return train_evaluate(df, features, "Lasso", strategy_name, Lasso, alpha)


def train_evaluate_ridge(df, features, strategy_name, alpha=1.0):
    """
    Train and evaluate the model using Ridge Regression.

    Parameters:
    df: The cleaned DataFrame.
    features: List of feature column names.
    strategy_name: Name of the strategy for logging.
    alpha: Regularization strength.

    Returns:
    Dictionary containing validation and test cMSE.
    """
    return train_evaluate(df, features, "Ridge", strategy_name, Ridge, alpha)


def select_best_strategy(strategy_results):
    """
    Select the best strategy based on the lowest Test cMSE.

    Parameters:
    strategy_results: List containing cMSE results for each strategy.

    Returns:
    The best strategy's details.
    """
    # Convert a list of dicts to DataFrame for easier manipulation
    results_df = pd.DataFrame(strategy_results)
    print("\n--- Strategy Performance ---")
    print(results_df)

    # Find the strategy with the minimum Test cMSE
    best_strategy = results_df.loc[results_df['test_cMSE'].idxmin()]
    print(f"\nBest Strategy Selected: {best_strategy['strategy']} with Test cMSE: {best_strategy['test_cMSE']:.4f}")

    return best_strategy
