import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from model import prepare_submission, SUBMISSION_EXTENSION, PREDICTION_COLUMN_NAME, plot_y_yhat, load_and_preprocess, \
    strategy_drop_rows, TRAIN_FILE_PATH, strategy_impute_missing, TEST_FILE_PATH, DATA_FILE_BASE_PATH, error_metric, \
    get_next_submission_filename

SUBMISSION_PREFIX = "knn-submission"

def plot_k_precision(train_mse_list, val_mse_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), train_mse_list, label="Training MSE", marker='o')
    plt.plot(range(1, 16), val_mse_list, label="Validation MSE", marker='o')

    min_val_mse = min(val_mse_list)
    min_k = val_mse_list.index(min_val_mse) + 1
    plt.axhline(y=min_val_mse, color='r', linestyle='--', label=f'Lowest Validation MSE = {min_val_mse:.4f} at K:{min_k}')

    plt.title('MSE vs K Value for Training and Validation Sets')
    plt.xlabel('K Value')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_evaluate_knn(df_train, df_validate, df_test, features, strategy_name, k):
    """
    Train and evaluate a KNN Regression model.

    Parameters:
    df_train (DataFrame): Training dataset.
    df_validate (DataFrame): Validation dataset.
    df_test (DataFrame): Test dataset.
    features (list): List of feature column names.
    strategy_name (str): Name of the preprocessing strategy.
    k (int): Number of neighbors for KNN.

    Returns:
    dict: Contains strategy name, k value, validation MSE, test MSE, and time taken.
    """
    print(f"\n--- Training and Evaluating KNN Regression (K={k}) for {strategy_name} ---")

    # Extract feature matrices and target vectors
    x_train = df_train[features].values
    y_train = df_train[PREDICTION_COLUMN_NAME].values
    x_validate = df_validate[features].values
    y_validate = df_validate[PREDICTION_COLUMN_NAME].values
    x_test = df_test[features].values
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_test = df_test["Censored"].values  # If needed for specialized metrics

    # Initialize KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=k)

    # Start timing
    start_time = time.time()

    # Train the model
    knn.fit(x_train, y_train)
    print(f"KNN Regression model with K={k} trained successfully.")

    # End timing
    elapsed_time = time.time() - start_time
    print(f"Time taken for training KNN (K={k}): {elapsed_time:.4f} seconds")

    # Make predictions
    y_train_pred = knn.predict(x_train)
    y_validate_pred = knn.predict(x_validate)
    y_test_pred = knn.predict(x_test)

    # Calculate MSE
    train_mse = error_metric(y_train, y_train_pred, c_test)
    validate_mse = error_metric(y_validate, y_validate_pred, c_test)
    test_mse = error_metric(y_test, y_test_pred, c_test)

    print(f"{strategy_name} - KNN Regression (K={k}) - Training MSE: {train_mse:.4f}")
    print(f"{strategy_name} - KNN Regression (K={k}) - Validation MSE: {validate_mse:.4f}")
    print(f"{strategy_name} - KNN Regression (K={k}) - Test MSE: {test_mse:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_validate), y_validate_pred,
                f'{strategy_name} - KNN Regression (K={k}) - Validation Predicted vs Actual')

    # Return the results
    return {
        'strategy': f"{strategy_name} - KNN (K={k})",
        'training_MSE': train_mse,
        'validation_MSE': validate_mse,
        'test_MSE': test_mse,
        'time_seconds': elapsed_time
    }


if __name__ == "__main__":
    # ===============================
    # 1. Load Training Data
    # ===============================
    print("Loading training data...")
    df_original = load_and_preprocess(TRAIN_FILE_PATH)
    print("Original DataFrame shape:", df_original.shape)

    # ===============================
    # 2. Apply Preprocessing Strategies
    # ===============================
    print("\nApplying Preprocessing Strategies...")

    # Strategy 1: Drop Rows with Missing Data
    print("\n--- Applying Strategy 1: Drop Rows with Missing Data ---")
    df_strategy1, features_strategy1 = strategy_drop_rows(df_original.copy())
    print(f"Strategy 1 - Number of data points after dropping rows: {len(df_strategy1)}")

    # Strategy 2: Impute Missing Values with Mean
    print("\n--- Applying Strategy 2: Impute Missing Values ---")
    df_strategy2, features_strategy2 = strategy_impute_missing(df_original.copy())
    print(f"Strategy 2 - Number of data points after imputation: {len(df_strategy2)}")

    # ===============================
    # 3. Train and Evaluate KNN Models
    # ===============================
    print("\nTraining and Evaluating KNN Regression Models...")

    # Initialize a list to store results
    strategy_results = []

    # Define range of K values to evaluate
    k_values = range(1, 16)

    # ---------------------------------
    # Strategy 1: Drop Rows
    # ---------------------------------
    print("\n=== Strategy 1: Drop Rows ===")

    for k in k_values:
        result_knn_s1 = train_evaluate_knn(
            df_train=df_strategy1,
            df_validate=df_strategy1,  # Typically, you should have separate validation
            df_test=df_strategy1,  # And separate test sets; adjust as needed
            features=features_strategy1,
            strategy_name="Strategy 1: Drop Rows",
            k=k
        )
        strategy_results.append(result_knn_s1)

    train_mse_list_s1 = [result['training_MSE'] for result in strategy_results]
    val_mse_list_s1 = [result['validation_MSE'] for result in strategy_results]
    plot_k_precision(train_mse_list_s1, val_mse_list_s1)

    # ---------------------------------
    # Strategy 2: Impute Missing Values
    # ---------------------------------
    print("\n=== Strategy 2: Impute Missing Values ===")

    for k in k_values:
        result_knn_s2 = train_evaluate_knn(
            df_train=df_strategy2,
            df_validate=df_strategy2,  # Typically, you should have separate validation
            df_test=df_strategy2,  # And separate test sets; adjust as needed
            features=features_strategy2,
            strategy_name="Strategy 2: Impute Missing Values",
            k=k
        )
        strategy_results.append(result_knn_s2)

    train_mse_list_s2 = [result['training_MSE'] for result in strategy_results]
    val_mse_list_s2 = [result['validation_MSE'] for result in strategy_results]
    plot_k_precision(train_mse_list_s2, val_mse_list_s2)

    # ===============================
    # 4. Select the Best KNN Model
    # ===============================
    print("\nSelecting the Best KNN Model Based on Validation MSE...")

    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(strategy_results)
    print("\n--- KNN Model Performance ---")
    print(results_df[['strategy', 'training_MSE', 'validation_MSE', 'test_MSE', 'time_seconds']])

    # Find the model with the lowest validation MSE
    best_model_idx = results_df['validation_MSE'].idxmin()
    best_model_details = results_df.loc[best_model_idx]
    print(
        f"\nBest KNN Model: {best_model_details['strategy']} with Validation MSE: {best_model_details['validation_MSE']:.4f}")

    # Extract the best K value and strategy
    best_strategy, best_k = best_model_details['strategy'].split(" - ")
    best_k = int(best_k.strip("K=").strip(")"))

    # ===============================
    # 5. Retrain the Best KNN Model on Entire Data
    # ===============================
    print("\nRetraining the Best KNN Model on the Entire Cleaned Dataset...")

    if "Strategy 1: Drop Rows" in best_model_details['strategy']:
        selected_features = features_strategy1
        selected_df = df_strategy1
    else:
        selected_features = features_strategy2
        selected_df = df_strategy2

    # Split the entire data into train and validation if needed
    # Here, we assume the entire data is used for training; adjust as per requirements
    X_full = selected_df[selected_features].values
    y_full = selected_df[PREDICTION_COLUMN_NAME].values

    # Initialize and train the best KNN model
    best_knn = KNeighborsRegressor(n_neighbors=best_k)
    start_time = time.time()
    best_knn.fit(X_full, y_full)
    elapsed_time = time.time() - start_time
    print(f"Best KNN Model with K={best_k} trained on the entire dataset in {elapsed_time:.4f} seconds.")

    # ===============================
    # 6. Load and Preprocess Kaggle Test Data
    # ===============================
    print("\nLoading Kaggle Test Data...")
    kaggle_test_data = load_and_preprocess(TEST_FILE_PATH)
    print("Kaggle Test DataFrame shape:", kaggle_test_data.shape)

    # Apply the same preprocessing strategy to test data
    if "Strategy 1: Drop Rows" in best_model_details['strategy']:
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
    X_kaggle_test = kaggle_test_clean[selected_features].values

    # ===============================
    # 7. Make Predictions on Test Data
    # ===============================
    print("\nMaking Predictions on Kaggle Test Data...")
    y_kaggle_pred = best_knn.predict(X_kaggle_test)

    # ===============================
    # 8. Prepare and Save Submission
    # ===============================
    print("\nPreparing Submission File...")

    # Generate the next submission filename
    submission_filename = get_next_submission_filename(
        base_path=DATA_FILE_BASE_PATH,
        prefix=SUBMISSION_PREFIX,
        extension=SUBMISSION_EXTENSION
    )
    print(f"Generated Submission Filename: {submission_filename}")

    # Prepare and save the submission
    prepare_submission(kaggle_test_clean, y_kaggle_pred, submission_filename)
    print("Submission process completed successfully.")