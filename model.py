import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
DATA_FILE_BASE_PATH = "data/"
FEATURE_COLUMN_NAMES = ["Age", "Gender", "Stage", "GeneticRisk", "TreatmentType", "ComorbidityIndex",
                        "TreatmentResponse"]
TARGET_COLUMN_NAMES = ["SurvivalTime", "Censored"]


def load_and_preprocess(filename):
    """
    Load and preprocess the dataset from a CSV file.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(filename)
    if df.columns[0].lower() != "id":
        df = df.rename(columns={df.columns[0]: "id"})
    return df


def visualize_missing_data(df, title_prefix=""):
    """
    Visualize missing data in the DataFrame using various plots.

    Parameters:
    df (pd.DataFrame): The DataFrame to visualize.
    title_prefix (str): A prefix for plot titles to differentiate strategies.
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
    df (pd.DataFrame): The DataFrame to visualize.
    title (str): The title for the visualization.
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
    df (pd.DataFrame): The DataFrame to split.
    train_size (float): Proportion of the dataset to include in the train split.
    validate_size (float): Proportion of the dataset to include in the validation split.
    test_size (float): Proportion of the dataset to include in the test split.
    seed (int): Random seed for reproducibility.

    Returns:
    tuple: Training, validation, and test DataFrames.
    """
    assert train_size + validate_size + test_size == 1.0, "Train, validate, and test sizes must sum to 1."

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end = int(train_size * len(df_shuffled))
    validate_end = train_end + int(validate_size * len(df_shuffled))

    df_train = df_shuffled.iloc[:train_end]
    df_validate = df_shuffled.iloc[train_end:validate_end]
    df_test = df_shuffled.iloc[validate_end:]

    return df_train, df_validate, df_test


def build_and_train_model(x_train, y_train):
    """
    Build and train a linear regression model using a pipeline with standardization.

    Parameters:
    x_train (pd.DataFrame): The feature matrix for training.
    y_train (pd.Series): The target vector for training.

    Returns:
    sklearn.pipeline.Pipeline: The trained model pipeline.
    """
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    return model_pipeline


def error_metric(y, y_hat, c):
    """
    Calculate the censored Mean Squared Error (cMSE).

    Parameters:
    y (pd.Series): The true survival time.
    y_hat (pd.Series): The predicted survival time.
    c (pd.Series): The censored variable.

    Returns:
    float: The cMSE value.
    """
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]


def plot_y_yhat(y_val, y_val_pred, title):
    """
    Plot the actual vs. predicted values for the validation set.

    Parameters:
    y_val (pd.Series): The actual values.
    y_val_pred (pd.Series): The predicted values.
    title (str): The title for the plot.
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
    test_df (pd.DataFrame): The test DataFrame containing 'id'.
    predictions (np.array): The predicted survival times.
    filename (str): The filename for the submission CSV.
    """
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        '0': predictions  # Changed from 'SurvivalTime' to '0' to match sample submission
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def clean_dataframe(df, feature_columns):
    """
    Clean the DataFrame by dropping rows with missing SurvivalTime or censored data points.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    feature_columns (list): List of feature column names.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    list: The list of features used.
    """
    # Drop rows with missing SurvivalTime or censored data points
    df_clean = df.dropna(subset=["SurvivalTime", "Censored"])
    df_clean = df_clean[df_clean["Censored"] == 0]  # Keep only uncensored data

    # Update the feature list after cleaning
    existing_features = [col for col in feature_columns if col in df_clean.columns]
    return df_clean, existing_features


def strategy_drop_rows(df):
    """
    Strategy 1: Drop rows with any missing data and drop censored data points.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame after dropping rows.
    list: The list of features used.
    """
    print("\n--- Strategy 1: Drop Rows with Missing Data ---")

    # Visualize missing data before cleaning
    visualize_missing_data(df, title_prefix="Strategy 1")

    # Drop rows with any missing values in feature columns
    df_clean = df.dropna(subset=FEATURE_COLUMN_NAMES)
    print(f"DataFrame shape after dropping rows with missing feature values: {df_clean.shape}")

    # Clean the DataFrame by dropping censored data points and missing SurvivalTime
    df_clean, existing_features = clean_dataframe(df_clean, FEATURE_COLUMN_NAMES)
    print(f"Features after dropping rows: {existing_features}")

    return df_clean, existing_features


def strategy_impute_missing(df):
    """
    Strategy 2: Impute missing values with the mean and drop censored data points.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame after imputation and dropping rows.
    list: The list of features used.
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

    # Clean the DataFrame by dropping censored data points and missing SurvivalTime
    df_imputed, existing_features = clean_dataframe(df_imputed, FEATURE_COLUMN_NAMES)
    print(f"Features after imputation: {existing_features}")

    return df_imputed, existing_features


def validate_model(model, df_validate, features, strategy_name, dataset_type="Validation"):
    """
    Validate the model and calculate cMSE for the dataset.

    Parameters:
    model (sklearn.pipeline.Pipeline): The trained model pipeline.
    df_validate (pd.DataFrame): The validation/test DataFrame.
    features (list): List of feature column names.
    strategy_name (str): Name of the strategy for logging.
    dataset_type (str): Type of the dataset (Validation/Test).

    Returns:
    float: The cMSE for the dataset.
    """
    X = df_validate[features]
    y = df_validate["SurvivalTime"]
    y_pred = model.predict(X)

    # Calculate cMSE
    censored = df_validate["Censored"]  # Should be all 0 for uncensored data
    cMSE = error_metric(y, y_pred, censored)
    print(f"{strategy_name} - {dataset_type} cMSE: {cMSE:.4f}")

    # Plot predicted vs actual
    plot_y_yhat(y, y_pred, f'{strategy_name} - {dataset_type} Predicted vs Actual')

    return cMSE


def train_evaluate(df, features, strategy_name):
    """
    Train and evaluate the model, then return cMSE.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame.
    features (list): List of feature column names.
    strategy_name (str): Name of the strategy for logging.

    Returns:
    dict: Dictionary containing validation and test cMSE.
    """
    print(f"\n--- Training and Evaluating Model for {strategy_name} ---")

    # Split the data into training, validation, and test sets
    df_train, df_validate, df_test = train_validate_split(df, train_size=0.6, validate_size=0.2, test_size=0.2,
                                                          seed=42)
    print(f"{strategy_name} - Training set: {len(df_train)} samples")
    print(f"{strategy_name} - Validation set: {len(df_validate)} samples")
    print(f"{strategy_name} - Test set: {len(df_test)} samples")

    # Define the feature matrix and target vector for training
    X_train = df_train[features]
    y_train = df_train["SurvivalTime"]

    # Train the model
    model = build_and_train_model(X_train, y_train)

    # Validate the model
    val_cMSE = validate_model(model, df_validate, features, strategy_name, dataset_type="Validation")

    # Evaluate on the hold-out test set
    test_cMSE = validate_model(model, df_test, features, strategy_name, dataset_type="Test")

    return {
        'strategy': strategy_name,
        'validation_cMSE': val_cMSE,
        'test_cMSE': test_cMSE
    }


def select_best_strategy(strategy_results):
    """
    Select the best strategy based on the lowest Test cMSE.

    Parameters:
    strategy_results (list of dict): List containing cMSE results for each strategy.

    Returns:
    dict: The best strategy's details.
    """
    # Convert list of dicts to DataFrame for easier manipulation
    results_df = pd.DataFrame(strategy_results)
    print("\n--- Strategy Performance ---")
    print(results_df)

    # Find the strategy with the minimum Test cMSE
    best_strategy = results_df.loc[results_df['test_cMSE'].idxmin()]
    print(f"\nBest Strategy Selected: {best_strategy['strategy']} with Test cMSE: {best_strategy['test_cMSE']:.4f}")

    return best_strategy


if __name__ == "__main__":
    # Load training dataset
    df_original = load_and_preprocess(DATA_FILE_BASE_PATH + "train_data.csv")
    print("Original DataFrame shape:", df_original.shape)

    # Implement Strategy 1: Drop rows with any missing data
    df_strategy1, features_strategy1 = strategy_drop_rows(df_original.copy())
    num_points_strategy1 = len(df_strategy1)
    print(f"Number of data points after Strategy 1: {num_points_strategy1}")

    # Implement Strategy 2: Impute missing values
    df_strategy2, features_strategy2 = strategy_impute_missing(df_original.copy())
    num_points_strategy2 = len(df_strategy2)
    print(f"Number of data points after Strategy 2: {num_points_strategy2}")

    # Train and evaluate for Strategy 1: Drop rows
    result_s1 = train_evaluate(df_strategy1, features_strategy1, "Strategy 1: Drop Rows")

    # Train and evaluate for Strategy 2: Impute missing values
    result_s2 = train_evaluate(df_strategy2, features_strategy2, "Strategy 2: Impute Missing Values")

    # Collect all strategy results
    strategy_results = [result_s1, result_s2]

    # Compare the results and select the best strategy
    best_strategy = select_best_strategy(strategy_results)
    selected_strategy_name = best_strategy['strategy']
    selected_test_cMSE = best_strategy['test_cMSE']

    # Determine which features to use based on the selected strategy
    if selected_strategy_name == "Strategy 1: Drop Rows":
        selected_features = features_strategy1
        selected_df = df_strategy1
    else:
        selected_features = features_strategy2
        selected_df = df_strategy2

    # Load Kaggle test data
    kaggle_test_data = load_and_preprocess(DATA_FILE_BASE_PATH + "test_data.csv")
    print("\nKaggle Test DataFrame shape:", kaggle_test_data.shape)

    # Preprocess Kaggle test data based on the selected strategy
    if selected_strategy_name == "Strategy 1: Drop Rows":
        # Strategy 1: Drop rows with any missing feature values
        kaggle_test_clean = kaggle_test_data.dropna(subset=selected_features)
        print(f"Dropped rows with missing feature values in Kaggle test data: {kaggle_test_clean.shape}")
    else:
        # Strategy 2: Impute missing values with mean
        kaggle_test_clean = kaggle_test_data.copy()
        for col in selected_features:
            if col in kaggle_test_clean.columns:
                mean_value = selected_df[col].mean()
                kaggle_test_clean[col] = kaggle_test_clean[col].fillna(mean_value)
                print(f"Imputed missing values in '{col}' with mean value {mean_value:.2f} in Kaggle test data")

        # Drop any remaining rows with missing values in the selected features (if any)
        kaggle_test_clean = kaggle_test_clean.dropna(subset=selected_features)
        print(f"Kaggle Test DataFrame shape after imputation: {kaggle_test_clean.shape}")

    # Ensure that the test data has the required features
    missing_features = [feature for feature in selected_features if feature not in kaggle_test_clean.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing in the Kaggle test data: {missing_features}")

    # Extract features for prediction
    X_kaggle_test = kaggle_test_clean[selected_features]

    # Retrain the model on the entire cleaned training data for the selected strategy
    model_selected = build_and_train_model(selected_df[selected_features], selected_df["SurvivalTime"])
    print(f"\nRetrained model using {selected_strategy_name}.")

    # Predict on the Kaggle test data
    y_kaggle_pred = model_selected.predict(X_kaggle_test)

    # Prepare and save the submission file
    submission_filename = DATA_FILE_BASE_PATH + 'baseline-submission-01.csv'
    prepare_submission(kaggle_test_clean, y_kaggle_pred, submission_filename)
