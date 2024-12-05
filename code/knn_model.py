import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool

from model import (
    prepare_submission, SUBMISSION_EXTENSION, PREDICTION_COLUMN_NAME, plot_y_yhat, load_and_preprocess,
    strategy_drop_rows, TRAIN_FILE_PATH, strategy_impute_missing, TEST_FILE_PATH, DATA_FILE_BASE_PATH,
    error_metric, get_next_submission_filename, train_validate_split, impute_missing_values
)

# Constants
SUBMISSION_PREFIX = "knn-submission"
HANDLING_MISSING_SUBMISSION_PREFIX = "handling-missing-submission"
FEATURE_COLUMN_NAMES = [
    "Age", "Gender", "Stage", "GeneticRisk", "TreatmentType",
    "ComorbidityIndex", "TreatmentResponse"
]
TARGET_COLUMNS = ["SurvivalTime", "Censored"]
ID_COLUMN_NAME = "id"


def plot_k_precision(train_mse_list, val_mse_list):
    """
    Plot the MSE against different K values for training and validation sets.

    Parameters:
    train_mse_list (list): List of training MSE values for different K values.
    val_mse_list (list): List of validation MSE values for different K values.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), train_mse_list, label="Training MSE", marker='o')
    plt.plot(range(1, 16), val_mse_list, label="Validation MSE", marker='o')

    min_val_mse = min(val_mse_list)
    min_k = val_mse_list.index(min_val_mse) + 1
    plt.axhline(y=min_val_mse, color='r', linestyle='--',
                label=f'Lowest Validation MSE = {min_val_mse:.4f} at K:{min_k}')

    plt.title('MSE vs K Value for Training and Validation Sets')
    plt.xlabel('K Value')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_k_time(time_list):
    """
    Plot the time taken for training and predicting against different K values.

    Parameters:
    time_list (list): List of time taken for training and predicting for different K values

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), time_list, label="Time Taken", marker='o', color='r')
    plt.title('Time Taken vs K Value for Training and Predicting')
    plt.xlabel('K Value')
    plt.ylabel('Time (seconds)')
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

    # Extract feature matrices and target vectors as DataFrames
    x_train = df_train[features]
    y_train = df_train[PREDICTION_COLUMN_NAME].values
    c_train = df_train["Censored"].values

    x_validate = df_validate[features]
    y_validate = df_validate[PREDICTION_COLUMN_NAME].values
    c_validate = df_validate["Censored"].values

    x_test = df_test[features]
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_test = df_test["Censored"].values

    # Initialize KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=k)

    # Start timing
    start_timestamp = time.time()

    # Train the model and measure time taken
    knn.fit(x_train, y_train)
    time_taken = time.time() - start_timestamp
    print(f"KNN Regression model with K={k} trained successfully.")
    print(f"Time taken for training KNN (K={k}): {time_taken:.4f} seconds")

    # Make predictions
    y_train_pred = knn.predict(x_train)
    y_validate_pred = knn.predict(x_validate)
    y_test_pred = knn.predict(x_test)

    # Calculate MSE using the corresponding censoring indicators
    train_mse = error_metric(y_train, y_train_pred, c_train)
    validate_mse = error_metric(y_validate, y_validate_pred, c_validate)
    test_mse = error_metric(y_test, y_test_pred, c_test)

    print(f"{strategy_name} - KNN Regression (K={k}) - Training MSE: {train_mse:.4f}")
    print(f"{strategy_name} - KNN Regression (K={k}) - Validation MSE: {validate_mse:.4f}")
    print(f"{strategy_name} - KNN Regression (K={k}) - Test MSE: {test_mse:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_validate), y_validate_pred,
                f'{strategy_name} - KNN Regression (K={k}) - Validation Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - KNN (K={k})",
        'training_MSE': train_mse,
        'validation_MSE': validate_mse,
        'test_MSE': test_mse,
        'time_seconds': time_taken
    }


def train_all_knn(df_train, df_validate, df_test, features, strategy_name):
    """
    Train and evaluate multiple KNN Regression models with different K values.

    Parameters:
    df_train (DataFrame): Training dataset.
    df_validate (DataFrame): Validation dataset.
    df_test (DataFrame): Test dataset.
    features (list): List of feature column names.
    strategy_name (str): Name of the preprocessing strategy.

    Returns:
    list: Contains results for all KNN models for a single strategy.
    """
    # Define range of K values to evaluate
    k_values = range(1, 16)

    results = []
    train_mse_list = []
    val_mse_list = []
    elapsed_time_list = []

    for k_value in k_values:
        result = train_evaluate_knn(
            df_train=df_train,
            df_validate=df_validate,
            df_test=df_test,
            features=features,
            strategy_name=strategy_name,
            k=k_value
        )
        results.append(result)
        train_mse_list.append(result['training_MSE'])
        val_mse_list.append(result['validation_MSE'])
        elapsed_time_list.append(result['time_seconds'])

    plot_k_precision(train_mse_list, val_mse_list)
    plot_k_time(elapsed_time_list)

    return results


def train_evaluate_hist_gradient_boosting(df_train, df_validate, df_test, features, strategy_name, learning_rate=0.1, max_iter=100):
    """
    Train and evaluate the HistGradientBoostingRegressor model.

    Parameters:
    df_train (DataFrame): Training dataset.
    df_validate (DataFrame): Validation dataset.
    df_test (DataFrame): Test dataset.
    features (list): List of feature column names.
    strategy_name (str): Name of the preprocessing strategy.
    learning_rate (float): Learning rate for the model.
    max_iter (int): Maximum number of iterations (trees).

    Returns:
    dict: Contains strategy name, validation cMSE, and test cMSE.
    """
    print(f"\n--- Training and Evaluating HistGradientBoostingRegressor for {strategy_name} ---")

    # Define the feature matrix and target vector for training
    x_train = df_train[features]
    y_train = df_train[PREDICTION_COLUMN_NAME].values
    c_train = df_train["Censored"].values

    # Initialize the HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(
        loss='squared_error',  # or 'least_squares'
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42
    )

    # Train the model
    start_time = time.time()
    hgb.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"HistGradientBoostingRegressor trained successfully in {elapsed_time:.4f} seconds.")

    # Predict on validation set
    y_val_pred = hgb.predict(df_validate[features])
    c_val = df_validate["Censored"].values
    y_val = df_validate[PREDICTION_COLUMN_NAME].values
    c_mse_val = error_metric(y_val, y_val_pred, c_val)
    print(f"{strategy_name} - HistGradientBoostingRegressor - Validation cMSE: {c_mse_val:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_val), y_val_pred, f'{strategy_name} - HistGradientBoostingRegressor - Validation Predicted vs Actual')

    # Predict on test set
    y_test_pred = hgb.predict(df_test[features])
    c_test = df_test["Censored"].values
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_mse_test = error_metric(y_test, y_test_pred, c_test)
    print(f"{strategy_name} - HistGradientBoostingRegressor - Test cMSE: {c_mse_test:.4f}")

    # Plot predicted vs. actual for the test set
    plot_y_yhat(pd.Series(y_test), y_test_pred, f'{strategy_name} - HistGradientBoostingRegressor - Test Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - HistGradientBoostingRegressor",
        'validation_cMSE': c_mse_val,
        'test_cMSE': c_mse_test
    }


def train_evaluate_catboost_aft(df_train, df_validate, df_test, features, strategy_name, iterations=15, learning_rate=0.1, depth=6):
    """
    Train and evaluate the CatBoostRegressor model.

    Parameters:
    df_train (DataFrame): Training dataset.
    df_validate (DataFrame): Validation dataset.
    df_test (DataFrame): Test dataset.
    features (list): List of feature column names.
    strategy_name (str): Name of the preprocessing strategy.
    iterations (int): Number of boosting iterations.
    learning_rate (float): Learning rate for the model.
    depth (int): Depth of the trees.

    Returns:
    dict: Contains strategy name, validation cMSE, and test cMSE.
    """
    print(f"\n--- Training and Evaluating CatBoostRegressor for {strategy_name} ---")

    # Prepare the feature matrix and target vector for training
    X_train = df_train[features]
    y_train = df_train[PREDICTION_COLUMN_NAME].values

    X_validate = df_validate[features]
    y_validate = df_validate[PREDICTION_COLUMN_NAME].values
    c_validate = df_validate["Censored"].values  # Censoring indicator for validation

    X_test = df_test[features]
    y_test = df_test[PREDICTION_COLUMN_NAME].values
    c_test = df_test["Censored"].values  # Censoring indicator for test

    # Initialize CatBoostRegressor with a supported loss function
    catboost_model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function='RMSE',  # Supported loss function
        eval_metric='RMSE',
        random_seed=42,
        verbose=100
    )

    # Train the model
    start_time = time.time()
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_validate, y_validate),
        early_stopping_rounds=5
    )
    elapsed_time = time.time() - start_time
    print(f"CatBoostRegressor trained successfully in {elapsed_time:.4f} seconds.")

    # Predict on validation set
    y_val_pred = catboost_model.predict(X_validate)
    mse_val = error_metric(y_validate, y_val_pred, c_validate)  # Pass censoring information
    print(f"{strategy_name} - CatBoostRegressor - Validation cMSE: {mse_val:.4f}")

    # Plot predicted vs. actual for the validation set
    plot_y_yhat(pd.Series(y_validate), y_val_pred, f'{strategy_name} - CatBoostRegressor - Validation Predicted vs Actual')

    # Predict on test set
    y_test_pred = catboost_model.predict(X_test)
    mse_test = error_metric(y_test, y_test_pred, c_test)  # Pass censoring information
    print(f"{strategy_name} - CatBoostRegressor - Test cMSE: {mse_test:.4f}")

    # Plot predicted vs. actual for the test set
    plot_y_yhat(pd.Series(y_test), y_test_pred, f'{strategy_name} - CatBoostRegressor - Test Predicted vs Actual')

    return {
        'strategy': f"{strategy_name} - CatBoostRegressor",
        'validation_cMSE': mse_val,
        'test_cMSE': mse_test
    }



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

    # Split Strategy 1 data into Train, Validate, Test
    df_train_s1, df_validate_s1, df_test_s1 = train_validate_split(df_strategy1)
    print(f"Strategy 1 - Training set: {df_train_s1.shape}")
    print(f"Strategy 1 - Validation set: {df_validate_s1.shape}")
    print(f"Strategy 1 - Test set: {df_test_s1.shape}")

    # Strategy 2: Impute Missing Values with Mean (Baseline)
    print("\n--- Applying Strategy 2: Impute Missing Values (Mean) ---")
    df_strategy2, features_strategy2 = strategy_impute_missing(df_original.copy())
    print(f"Strategy 2 - Number of data points after imputation: {len(df_strategy2)}")

    # Split Strategy 2 data into Train, Validate, Test
    df_train_s2, df_validate_s2, df_test_s2 = train_validate_split(df_strategy2)
    print(f"Strategy 2 - Training set: {df_train_s2.shape}")
    print(f"Strategy 2 - Validation set: {df_validate_s2.shape}")
    print(f"Strategy 2 - Test set: {df_test_s2.shape}")

    # ===============================
    # 3. Train and Evaluate KNN Models
    # ===============================
    print("\nTraining and Evaluating KNN Regression Models...")

    # Initialize a list to store results
    strategy_results = []

    # ---------------------------------
    # Strategy 1: Drop Rows
    # ---------------------------------
    print("\n=== Strategy 1: Drop Rows ===")

    strategy_1_results = train_all_knn(
        df_train=df_train_s1,
        df_validate=df_validate_s1,
        df_test=df_test_s1,
        features=features_strategy1,
        strategy_name="Strategy 1: Drop Rows"
    )
    strategy_results.extend(strategy_1_results)

    # ---------------------------------
    # Strategy 2: Impute Missing Values (Mean)
    # ---------------------------------
    print("\n=== Strategy 2: Impute Missing Values (Mean) ===")

    strategy_2_results = train_all_knn(
        df_train=df_train_s2,
        df_validate=df_validate_s2,
        df_test=df_test_s2,
        features=features_strategy2,
        strategy_name="Strategy 2: Impute Missing Values (Mean)"
    )
    strategy_results.extend(strategy_2_results)

    # ===============================
    # 4. Implement Advanced Imputation Techniques and Train Advanced Models
    # ===============================
    print("\n=== Task 3.2: Train Models That Do Not Require Imputation ===")

    # Define imputation strategies to experiment with
    advanced_imputation_strategies = ['median', 'most_frequent', 'knn', 'iterative']

    for strategy in advanced_imputation_strategies:
        strategy_name = f"Imputation: {strategy.capitalize()}"
        print(f"\n--- Applying {strategy_name} ---")

        # Apply the imputation strategy to the original data
        df_imputed = impute_missing_values(df_original.copy(), strategy=strategy)

        # Split the imputed data into Train, Validate, Test
        df_train, df_validate, df_test = train_validate_split(df_imputed)

        # Define feature columns (ensure they are present after imputation)
        features = [col for col in FEATURE_COLUMN_NAMES if col in df_imputed.columns]
        print(f"{strategy_name} - Features used: {features}")

        # ---------------------------------
        # 4.1. Train and Evaluate KNN Models for Advanced Imputation Strategies
        # ---------------------------------
        print(f"\n--- Training KNN Regression for {strategy_name} ---")
        knn_results = train_all_knn(
            df_train=df_train,
            df_validate=df_validate,
            df_test=df_test,
            features=features,
            strategy_name=strategy_name
        )
        strategy_results.extend(knn_results)

        # ---------------------------------
        # 4.2. Train and Evaluate HistGradientBoostingRegressor for Advanced Imputation Strategies
        # ---------------------------------
        print(f"\n--- Training HistGradientBoostingRegressor for {strategy_name} ---")
        hgb_result = train_evaluate_hist_gradient_boosting(
            df_train=df_train,
            df_validate=df_validate,
            df_test=df_test,
            features=features,
            strategy_name=strategy_name,
            learning_rate=0.1,
            max_iter=100
        )
        strategy_results.append(hgb_result)

        # ---------------------------------
        # 4.3. Train and Evaluate CatBoostRegressor with AFT for Advanced Imputation Strategies
        # ---------------------------------
        print(f"\n--- Training CatBoostRegressor (AFT) for {strategy_name} ---")
        catboost_result = train_evaluate_catboost_aft(
            df_train=df_train,
            df_validate=df_validate,
            df_test=df_test,
            features=features,
            strategy_name=strategy_name,
            iterations=1000,
            learning_rate=0.1,
            depth=6
        )
        strategy_results.append(catboost_result)

    # ===============================
    # 5. Compare All Imputation Strategies Including New Models
    # ===============================
    print("\nComparing All Imputation Strategies with Previous Approaches...")

    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(strategy_results)
    print("\n--- Model Performance Across All Strategies ---")
    print(results_df[['strategy', 'training_MSE', 'validation_MSE', 'test_MSE', 'time_seconds']])

    # ===============================
    # 6. Choose the Best Model Based on Validation MSE
    # ===============================
    print("\nSelecting the Best Model Based on Validation MSE...")

    # Find the model with the lowest validation MSE
    best_model_idx = results_df['validation_MSE'].idxmin()
    best_model_details = results_df.loc[best_model_idx]
    print(f"\nBest Model: {best_model_details['strategy']} with Validation cMSE: {best_model_details['validation_MSE']:.4f}")

    # Extract the best strategy name and model type
    # Since the strategy name includes both the preprocessing strategy and the model type,
    # you'll need to parse it accordingly.

    # Example strategy string: "Imputation: Median - CatBoostRegressor (AFT)"
    if " - " in best_model_details['strategy']:
        best_strategy_name, best_model_type = best_model_details['strategy'].split(" - ", 1)
    else:
        best_strategy_name = best_model_details['strategy']
        best_model_type = "Unknown Model"

    print(f"Best Strategy: {best_strategy_name}")
    print(f"Best Model Type: {best_model_type}")

    # ===============================
    # 7. Retrain the Best Model on Entire Data
    # ===============================
    print("\nRetraining the Best Model on the Entire Cleaned Dataset...")

    if "Strategy 1: Drop Rows" in best_strategy_name:
        selected_features = features_strategy1
        selected_df = df_strategy1
    elif "Strategy 2: Impute Missing Values (Mean)" in best_strategy_name:
        selected_features = features_strategy2
        selected_df = df_strategy2
    else:
        # For advanced imputation strategies
        # Extract the imputation strategy from the strategy name
        imputation_strategy = best_strategy_name.replace("Imputation: ", "").lower()
        print(f"Selected Strategy: Imputation using {imputation_strategy.capitalize()} Imputer")
        selected_df = impute_missing_values(df_original.copy(), strategy=imputation_strategy)
        selected_features = [col for col in FEATURE_COLUMN_NAMES if col in selected_df.columns]
        print(f"Selected Strategy: Imputation using {imputation_strategy.capitalize()} Imputer")

    # Depending on the best model type, retrain accordingly
    if "KNN" in best_model_type:
        # Reuse the KNN Regressor
        best_k = int(best_model_type.split("(K=")[-1].rstrip(")"))
        best_knn = KNeighborsRegressor(n_neighbors=best_k)
        start_time = time.time()
        best_knn.fit(selected_df[selected_features], selected_df[PREDICTION_COLUMN_NAME].values)
        elapsed_time = time.time() - start_time
        print(f"Best KNN Model with K={best_k} trained on the entire dataset in {elapsed_time:.4f} seconds.")
    elif "HistGradientBoostingRegressor" in best_model_type:
        # Reuse the HistGradientBoostingRegressor
        hgb = HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        start_time = time.time()
        hgb.fit(selected_df[selected_features], selected_df[PREDICTION_COLUMN_NAME].values)
        elapsed_time = time.time() - start_time
        print(f"Best HistGradientBoostingRegressor trained on the entire dataset in {elapsed_time:.4f} seconds.")
    elif "CatBoostRegressor" in best_model_type:
        # Reuse the CatBoostRegressor with AFT
        catboost_aft = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='AFT',
            eval_metric='CensoredRMSE',
            random_seed=42,
            verbose=100
        )
        # Prepare the entire dataset as training Pool
        train_pool = Pool(
            data=selected_df[features_strategy1 if "Strategy 1: Drop Rows" in best_strategy_name else features_strategy2],
            label=selected_df[PREDICTION_COLUMN_NAME].values,
            censor=selected_df["Censored"].values
        )
        start_time = time.time()
        catboost_aft.fit(train_pool)
        elapsed_time = time.time() - start_time
        print(f"Best CatBoostRegressor (AFT) trained on the entire dataset in {elapsed_time:.4f} seconds.")
    else:
        raise ValueError("Best model type is not recognized.")

    # ===============================
    # 8. Load and Preprocess Kaggle Test Data
    # ===============================
    print("\nLoading Kaggle Test Data...")
    kaggle_test_data = load_and_preprocess(TEST_FILE_PATH)
    print("Kaggle Test DataFrame shape:", kaggle_test_data.shape)

    # Apply the same preprocessing strategy to test data
    if "Strategy 1: Drop Rows" in best_strategy_name:
        # Strategy 1: Drop rows with any missing feature values
        kaggle_test_clean = kaggle_test_data.dropna(subset=selected_features)
        print(f"Dropped rows with missing feature values in Kaggle test data: {kaggle_test_clean.shape}")
    elif "Strategy 2: Impute Missing Values (Mean)" in best_strategy_name:
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
    else:
        # For advanced imputation strategies
        # Reapply the imputation strategy to the entire test data
        imputation_strategy = best_strategy_name.replace("Imputation: ", "").lower()
        print(f"Imputation Strategy: {imputation_strategy.capitalize()} Imputer")
        kaggle_test_clean = impute_missing_values(kaggle_test_data.copy(), strategy=imputation_strategy)
        print(f"Imputed missing values in Kaggle test data using {imputation_strategy.capitalize()} Imputer.")

        # Drop any remaining rows with missing values in the selected features (if any)
        kaggle_test_clean = kaggle_test_clean.dropna(subset=selected_features)
        print(f"Kaggle Test DataFrame shape after imputation: {kaggle_test_clean.shape}")

    # Ensure that the test data has the required features
    missing_features = [feature for feature in selected_features if feature not in kaggle_test_clean.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing in the Kaggle test data: {missing_features}")

    # Extract features for prediction
    X_kaggle_test = kaggle_test_clean[selected_features]

    # ===============================
    # 9. Make Predictions on Test Data
    # ===============================
    print("\nMaking Predictions on Kaggle Test Data...")
    if "KNN" in best_model_type:
        y_kaggle_pred = best_knn.predict(X_kaggle_test)
    elif "HistGradientBoostingRegressor" in best_model_type:
        y_kaggle_pred = hgb.predict(X_kaggle_test)
    elif "CatBoostRegressor" in best_model_type:
        y_kaggle_pred = catboost_aft.predict(X_kaggle_test)
    else:
        raise ValueError("Best model type is not recognized.")

    # ===============================
    # 10. Prepare and Save Submission
    # ===============================
    print("\nPreparing Submission File...")

    # Generate the next submission filename
    submission_filename = get_next_submission_filename(
        base_path=DATA_FILE_BASE_PATH,
        prefix=HANDLING_MISSING_SUBMISSION_PREFIX,
        extension=SUBMISSION_EXTENSION
    )
    print(f"Generated Submission Filename: {submission_filename}")

    # Prepare and save the submission
    prepare_submission(kaggle_test_clean, y_kaggle_pred, submission_filename)
    print("Submission process completed successfully.")