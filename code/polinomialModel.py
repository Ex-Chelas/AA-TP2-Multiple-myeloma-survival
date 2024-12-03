import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

from model import (
    load_and_preprocess, prepare_submission, select_best_strategy, strategy_impute_missing, strategy_drop_rows,
    TRAIN_FILE_PATH, train_validate_split, PREDICTION_COLUMN_NAME, TEST_FILE_PATH
)

def compute_censored_mse(y_true, y_pred, c):
    """
    Compute the censored Mean Squared Error (cMSE).
    """
    # Only consider non-censored data points
    mask = c == 0
    mse = np.mean((y_true[mask] - y_pred[mask]) ** 2)
    return mse

if __name__ == "__main__":
    # Define ranges for degrees and alphas
    degrees = [1, 2, 3, 4, 5]
    alphas = [0.01, 0.1, 1, 10, 100]

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
    # 3. Train and Evaluate Models
    # ===============================
    print("\nTraining and Evaluating Models...")

    # Initialize a list to store results
    strategy_results = []

    # ---------------------------------
    # Strategy 1: Drop Rows
    # ---------------------------------
    print("\n=== Strategy 1: Drop Rows ===")

    # Split data
    df_train_s1, df_validate_s1, df_test_s1 = train_validate_split(df_strategy1)

    # Initialize a DataFrame to store results
    results_s1 = []

    # ---------------------------
    # Linear Regression
    # ---------------------------
    print("\n--- Training Linear Regression for Strategy 1 ---")

    for degree in degrees:
        print(f"\n--- Training Linear Regression with degree {degree} for Strategy 1 ---")
        # Create pipeline with PolynomialFeatures
        model_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('scaler', StandardScaler()),  # Standardization
            ('regressor', LinearRegression())
        ])
        # Extract features and labels
        # Extract features and labels as DataFrames and Series (with feature names)
        X_train = df_train_s1[features_strategy1]
        y_train = df_train_s1[PREDICTION_COLUMN_NAME]
        c_train = df_train_s1["Censored"]

        X_validate = df_validate_s1[features_strategy1]
        y_validate = df_validate_s1[PREDICTION_COLUMN_NAME]
        c_validate = df_validate_s1["Censored"]

        X_test = df_test_s1[features_strategy1]
        y_test = df_test_s1[PREDICTION_COLUMN_NAME]
        c_test = df_test_s1["Censored"]

        # Fit the model
        model_pipeline.fit(X_train, y_train)

        # Predict
        y_validate_pred = model_pipeline.predict(X_validate)
        y_test_pred = model_pipeline.predict(X_test)

        # Compute cMSE
        cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
        cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

        # Store results
        result = {
            'strategy': "Strategy 1: Drop Rows",
            'model': "Linear Regression",
            'degree': degree,
            'alpha': None,
            'validate_cMSE': cMSE_validate,
            'test_cMSE': cMSE_test
        }
        print(f"Degree: {degree}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
        results_s1.append(result)

    # ---------------------------
    # Lasso Regression
    # ---------------------------
    print("\n--- Training Lasso Regression for Strategy 1 ---")

    for degree in degrees:
        for alpha in alphas:
            print(f"\n--- Training Lasso Regression with degree {degree}, alpha {alpha} for Strategy 1 ---")
            # Create pipeline
            model_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('scaler', StandardScaler()),  # Standardization
                ('regressor', Lasso(alpha=alpha, max_iter=10000))
            ])
            # Fit and evaluate
            # Extract features and labels
            X_train = df_train_s1[features_strategy1].values
            y_train = df_train_s1[PREDICTION_COLUMN_NAME].values
            c_train = df_train_s1["Censored"].values

            X_validate = df_validate_s1[features_strategy1].values
            y_validate = df_validate_s1[PREDICTION_COLUMN_NAME].values
            c_validate = df_validate_s1["Censored"].values

            X_test = df_test_s1[features_strategy1].values
            y_test = df_test_s1[PREDICTION_COLUMN_NAME].values
            c_test = df_test_s1["Censored"].values

            # Fit the model
            model_pipeline.fit(X_train, y_train)

            # Predict
            y_validate_pred = model_pipeline.predict(X_validate)
            y_test_pred = model_pipeline.predict(X_test)

            # Compute cMSE
            cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
            cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

            # Store results
            result = {
                'strategy': "Strategy 1: Drop Rows",
                'model': "Lasso Regression",
                'degree': degree,
                'alpha': alpha,
                'validate_cMSE': cMSE_validate,
                'test_cMSE': cMSE_test
            }
            print(f"Degree: {degree}, Alpha: {alpha}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
            results_s1.append(result)

    # ---------------------------
    # Ridge Regression
    # ---------------------------
    print("\n--- Training Ridge Regression for Strategy 1 ---")

    for degree in degrees:
        for alpha in alphas:
            print(f"\n--- Training Ridge Regression with degree {degree}, alpha {alpha} for Strategy 1 ---")
            # Create pipeline
            model_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('scaler', StandardScaler()),  # Standardization
                ('regressor', Ridge(alpha=alpha))
            ])
            # Fit and evaluate
            # Extract features and labels
            X_train = df_train_s1[features_strategy1].values
            y_train = df_train_s1[PREDICTION_COLUMN_NAME].values
            c_train = df_train_s1["Censored"].values

            X_validate = df_validate_s1[features_strategy1].values
            y_validate = df_validate_s1[PREDICTION_COLUMN_NAME].values
            c_validate = df_validate_s1["Censored"].values

            X_test = df_test_s1[features_strategy1].values
            y_test = df_test_s1[PREDICTION_COLUMN_NAME].values
            c_test = df_test_s1["Censored"].values

            # Fit the model
            model_pipeline.fit(X_train, y_train)

            # Predict
            y_validate_pred = model_pipeline.predict(X_validate)
            y_test_pred = model_pipeline.predict(X_test)

            # Compute cMSE
            cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
            cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

            # Store results
            result = {
                'strategy': "Strategy 1: Drop Rows",
                'model': "Ridge Regression",
                'degree': degree,
                'alpha': alpha,
                'validate_cMSE': cMSE_validate,
                'test_cMSE': cMSE_test
            }
            print(f"Degree: {degree}, Alpha: {alpha}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
            results_s1.append(result)

    # Convert results to DataFrame
    results_s1_df = pd.DataFrame(results_s1)

    # Similarly, perform for Strategy 2
    # ---------------------------------
    # Strategy 2: Impute Missing Values
    # ---------------------------------
    print("\n=== Strategy 2: Impute Missing Values ===")

    # Split data
    df_train_s2, df_validate_s2, df_test_s2 = train_validate_split(df_strategy2)

    # Initialize a DataFrame to store results
    results_s2 = []

    # ---------------------------
    # Linear Regression
    # ---------------------------
    print("\n--- Training Linear Regression for Strategy 2 ---")

    for degree in degrees:
        print(f"\n--- Training Linear Regression with degree {degree} for Strategy 2 ---")
        # Create pipeline with PolynomialFeatures
        model_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('scaler', StandardScaler()),  # Standardization
            ('regressor', LinearRegression())
        ])
        # Extract features and labels
        X_train = df_train_s2[features_strategy2].values
        y_train = df_train_s2[PREDICTION_COLUMN_NAME].values
        c_train = df_train_s2["Censored"].values

        X_validate = df_validate_s2[features_strategy2].values
        y_validate = df_validate_s2[PREDICTION_COLUMN_NAME].values
        c_validate = df_validate_s2["Censored"].values

        X_test = df_test_s2[features_strategy2].values
        y_test = df_test_s2[PREDICTION_COLUMN_NAME].values
        c_test = df_test_s2["Censored"].values

        # Fit the model
        model_pipeline.fit(X_train, y_train)

        # Predict
        y_validate_pred = model_pipeline.predict(X_validate)
        y_test_pred = model_pipeline.predict(X_test)

        # Compute cMSE
        cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
        cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

        # Store results
        result = {
            'strategy': "Strategy 2: Impute Missing Values",
            'model': "Linear Regression",
            'degree': degree,
            'alpha': None,
            'validate_cMSE': cMSE_validate,
            'test_cMSE': cMSE_test
        }
        print(f"Degree: {degree}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
        results_s2.append(result)

    # ---------------------------
    # Lasso Regression
    # ---------------------------
    print("\n--- Training Lasso Regression for Strategy 2 ---")

    for degree in degrees:
        for alpha in alphas:
            print(f"\n--- Training Lasso Regression with degree {degree}, alpha {alpha} for Strategy 2 ---")
            # Create pipeline
            model_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('scaler', StandardScaler()),  # Standardization
                ('regressor', Lasso(alpha=alpha, max_iter=10000))
            ])
            # Fit and evaluate
            # Extract features and labels
            X_train = df_train_s2[features_strategy2].values
            y_train = df_train_s2[PREDICTION_COLUMN_NAME].values
            c_train = df_train_s2["Censored"].values

            X_validate = df_validate_s2[features_strategy2].values
            y_validate = df_validate_s2[PREDICTION_COLUMN_NAME].values
            c_validate = df_validate_s2["Censored"].values

            X_test = df_test_s2[features_strategy2].values
            y_test = df_test_s2[PREDICTION_COLUMN_NAME].values
            c_test = df_test_s2["Censored"].values

            # Fit the model
            model_pipeline.fit(X_train, y_train)

            # Predict
            y_validate_pred = model_pipeline.predict(X_validate)
            y_test_pred = model_pipeline.predict(X_test)

            # Compute cMSE
            cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
            cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

            # Store results
            result = {
                'strategy': "Strategy 2: Impute Missing Values",
                'model': "Lasso Regression",
                'degree': degree,
                'alpha': alpha,
                'validate_cMSE': cMSE_validate,
                'test_cMSE': cMSE_test
            }
            print(f"Degree: {degree}, Alpha: {alpha}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
            results_s2.append(result)

    # ---------------------------
    # Ridge Regression
    # ---------------------------
    print("\n--- Training Ridge Regression for Strategy 2 ---")

    for degree in degrees:
        for alpha in alphas:
            print(f"\n--- Training Ridge Regression with degree {degree}, alpha {alpha} for Strategy 2 ---")
            # Create pipeline
            model_pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('scaler', StandardScaler()),  # Standardization
                ('regressor', Ridge(alpha=alpha))
            ])
            # Fit and evaluate
            # Extract features and labels
            X_train = df_train_s2[features_strategy2].values
            y_train = df_train_s2[PREDICTION_COLUMN_NAME].values
            c_train = df_train_s2["Censored"].values

            X_validate = df_validate_s2[features_strategy2].values
            y_validate = df_validate_s2[PREDICTION_COLUMN_NAME].values
            c_validate = df_validate_s2["Censored"].values

            X_test = df_test_s2[features_strategy2].values
            y_test = df_test_s2[PREDICTION_COLUMN_NAME].values
            c_test = df_test_s2["Censored"].values

            # Fit the model
            model_pipeline.fit(X_train, y_train)

            # Predict
            y_validate_pred = model_pipeline.predict(X_validate)
            y_test_pred = model_pipeline.predict(X_test)

            # Compute cMSE
            cMSE_validate = compute_censored_mse(y_validate, y_validate_pred, c_validate)
            cMSE_test = compute_censored_mse(y_test, y_test_pred, c_test)

            # Store results
            result = {
                'strategy': "Strategy 2: Impute Missing Values",
                'model': "Ridge Regression",
                'degree': degree,
                'alpha': alpha,
                'validate_cMSE': cMSE_validate,
                'test_cMSE': cMSE_test
            }
            print(f"Degree: {degree}, Alpha: {alpha}, Validate cMSE: {cMSE_validate:.4f}, Test cMSE: {cMSE_test:.4f}")
            results_s2.append(result)

    # Convert results to DataFrame
    results_s2_df = pd.DataFrame(results_s2)

    # Combine results
    all_results_df = pd.concat([results_s1_df, results_s2_df], ignore_index=True)

    # ===============================
    # 4. Select the Best Strategy
    # ===============================
    print("\nSelecting the Best Strategy Based on Test cMSE...")
    best_result = all_results_df.loc[all_results_df['test_cMSE'].idxmin()]
    selected_strategy_name = best_result['strategy']
    selected_model_type = best_result['model']
    selected_degree = best_result['degree']
    selected_alpha = best_result['alpha']
    selected_test_cMSE = best_result['test_cMSE']

    print(f"\nBest Model Selected: {selected_model_type}")
    print(f"Strategy: {selected_strategy_name}")
    print(f"Degree: {selected_degree}, Alpha: {selected_alpha}")
    print(f"Test cMSE: {selected_test_cMSE:.4f}")

    # ===============================
    # 5. Plotting cMSE for Degree/Alpha Combinations
    # ===============================

    # For models with alpha parameter (Lasso and Ridge Regression)
    for model_type in ['Lasso Regression', 'Ridge Regression']:
        plt.figure(figsize=(10, 6))
        for strategy in all_results_df['strategy'].unique():
            strat_df = all_results_df[
                (all_results_df['model'] == model_type) &
                (all_results_df['strategy'] == strategy)
                ]
            # For each degree, find the alpha with the lowest test cMSE
            best_cMSE_per_degree = strat_df.loc[strat_df.groupby('degree')['test_cMSE'].idxmin()]
            plt.plot(
                best_cMSE_per_degree['degree'],
                best_cMSE_per_degree['test_cMSE'],
                marker='o',
                label=f"{strategy}"
            )
        plt.title(f"{model_type} - Best Test cMSE vs Degree")
        plt.xlabel("Degree")
        plt.ylabel("Test cMSE")
        plt.legend(title='Strategy')
        plt.grid(True)
        plt.show()

    # For Linear Regression (no alpha parameter)
    model_type = 'Linear Regression'
    plt.figure(figsize=(10, 6))
    for strategy in all_results_df['strategy'].unique():
        strat_df = all_results_df[
            (all_results_df['model'] == model_type) &
            (all_results_df['strategy'] == strategy)
            ]
        plt.plot(
            strat_df['degree'],
            strat_df['test_cMSE'],
            marker='o',
            label=f"{strategy}"
        )
    plt.title(f"{model_type} - Test cMSE vs Degree")
    plt.xlabel("Degree")
    plt.ylabel("Test cMSE")
    plt.legend(title='Strategy')
    plt.grid(True)
    plt.show()

    # ===============================
    # 6. Retrain the Best Model on Entire Data
    # ===============================
    print("\nRetraining the Best Model on the Entire Cleaned Dataset...")

    # Determine which features and DataFrame to use
    if "Strategy 1: Drop Rows" in selected_strategy_name:
        selected_features = features_strategy1
        selected_df = df_strategy1
    else:
        selected_features = features_strategy2
        selected_df = df_strategy2

    # Load Kaggle test data
    print("\nLoading Kaggle Test Data...")
    kaggle_test_data = load_and_preprocess(TEST_FILE_PATH)
    print("Kaggle Test DataFrame shape:", kaggle_test_data.shape)

    # Preprocess Kaggle test data based on the selected strategy
    if "Strategy 1: Drop Rows" in selected_strategy_name:
        # Strategy 1: Drop rows with any missing feature values
        kaggle_test_clean = kaggle_test_data.dropna(subset=selected_features)
        print(f"Dropped rows with missing feature values in Kaggle test data: {kaggle_test_clean.shape}")
    else:
        # Strategy 2: Impute missing values with mean from training data
        column_means = df_original[selected_features].mean()
        kaggle_test_clean = kaggle_test_data.copy()
        for col in selected_features:
            if col in kaggle_test_clean.columns:
                mean_value = column_means[col]
                kaggle_test_clean[col] = kaggle_test_clean[col].fillna(mean_value)
                print(f"Imputed missing values in '{col}' with mean value {mean_value:.2f} from training data")

    # Ensure that the test data has the required features
    missing_features = [feature for feature in selected_features if feature not in kaggle_test_clean.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing in the Kaggle test data: {missing_features}")

    # Extract features for prediction
    X_kaggle_test = kaggle_test_clean[selected_features].values

    # Retrain the Best Model
    print(f"\nRetraining {selected_model_type} with degree {selected_degree} and alpha {selected_alpha} on the Entire Dataset...")
    # Create pipeline with selected degree and alpha
    if selected_model_type == "Linear Regression":
        model_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=selected_degree)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    elif selected_model_type == "Lasso Regression":
        model_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=selected_degree)),
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=selected_alpha, max_iter=10000))
        ])
    elif selected_model_type == "Ridge Regression":
        model_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=selected_degree)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=selected_alpha))
        ])
    else:
        raise ValueError("Unknown model type selected.")

    # Fit the model on the entire dataset
    X_full = selected_df[selected_features].values
    y_full = selected_df[PREDICTION_COLUMN_NAME].values
    model_pipeline.fit(X_full, y_full)
    print("Best model retrained successfully.")

    # Predict on Kaggle test data
    y_kaggle_pred = model_pipeline.predict(kaggle_test_clean[selected_features])

    # ===============================
    # 7. Prepare and Save Submission
    # ===============================
    print("\nPreparing Submission File...")

    # Prepare and save the submission
    submission_filename = "submission.csv"  # Replace with your desired filename
    prepare_submission(kaggle_test_clean, y_kaggle_pred, submission_filename)
    print("Submission process completed successfully.")
