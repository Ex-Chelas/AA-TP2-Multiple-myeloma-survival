import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import (
    load_and_preprocess, prepare_submission, select_best_strategy, strategy_impute_missing, strategy_drop_rows,
    TRAIN_FILE_PATH, train_evaluate_custom_gd, train_evaluate_lasso, train_evaluate_ridge, gradient_descent_c_mse,
    train_and_evaluate_model, train_validate_split, PREDICTION_COLUMN_NAME, TEST_FILE_PATH
)


def build_and_train_model(x_train, y_train):
    """
    Build and train a linear regression model using a pipeline with standardization.

    Parameters:
    x_train: The feature matrix for training.
    y_train: The target vector for training.

    Returns:
    The trained model pipeline.
    """
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    print("Scikit-learn Linear Regression model trained successfully.")
    return model_pipeline


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
    # 3. Train and Evaluate Models
    # ===============================
    print("\nTraining and Evaluating Models...")

    # Initialize a list to store results
    strategy_results = []

    # ---------------------------------
    # Strategy 1: Drop Rows
    # ---------------------------------
    print("\n=== Strategy 1: Drop Rows ===")

    # ---------------------------
    # Linear Regression
    # ---------------------------
    print("\n--- Training Linear Regression for Strategy 1 ---")
    # Split data
    df_train_s1, df_validate_s1, df_test_s1 = train_validate_split(df_strategy1)
    # Initialize model
    model_lr_s1 = LinearRegression()
    # Train and evaluate
    result_lr_s1 = train_and_evaluate_model(
        df_train=df_train_s1,
        df_validate=df_validate_s1,
        df_test=df_test_s1,
        features=features_strategy1,
        strategy_name="Strategy 1: Drop Rows",
        model=model_lr_s1,
        model_type="Linear Regression"
    )
    strategy_results.append(result_lr_s1)

    # ---------------------------
    # Lasso Regression
    # ---------------------------
    print("\n--- Training Lasso Regression for Strategy 1 ---")
    result_lasso_s1 = train_evaluate_lasso(
        df=df_strategy1,
        features=features_strategy1,
        strategy_name="Strategy 1: Drop Rows",
        alpha=0.1
    )
    strategy_results.append(result_lasso_s1)

    # ---------------------------
    # Ridge Regression
    # ---------------------------
    print("\n--- Training Ridge Regression for Strategy 1 ---")
    result_ridge_s1 = train_evaluate_ridge(
        df=df_strategy1,
        features=features_strategy1,
        strategy_name="Strategy 1: Drop Rows",
        alpha=1.0
    )
    strategy_results.append(result_ridge_s1)

    # ---------------------------
    # Custom Gradient Descent
    # ---------------------------
    print("\n--- Training Custom Gradient Descent for Strategy 1 ---")
    result_gd_s1 = train_evaluate_custom_gd(
        df=df_strategy1,
        features=features_strategy1,
        strategy_name="Strategy 1: Drop Rows",
        learning_rate=0.001,
        epochs=1000,
        tolerance=1e-6
    )
    strategy_results.append(result_gd_s1)

    # ---------------------------------
    # Strategy 2: Impute Missing Values
    # ---------------------------------
    print("\n=== Strategy 2: Impute Missing Values ===")

    # ---------------------------
    # Linear Regression
    # ---------------------------
    print("\n--- Training Linear Regression for Strategy 2 ---")
    # Split data
    df_train_s2, df_validate_s2, df_test_s2 = train_validate_split(df_strategy2)
    # Initialize model
    model_lr_s2 = LinearRegression()
    # Train and evaluate
    result_lr_s2 = train_and_evaluate_model(
        df_train=df_train_s2,
        df_validate=df_validate_s2,
        df_test=df_test_s2,
        features=features_strategy2,
        strategy_name="Strategy 2: Impute Missing Values",
        model=model_lr_s2,
        model_type="Linear Regression"
    )
    strategy_results.append(result_lr_s2)

    # ---------------------------
    # Lasso Regression
    # ---------------------------
    print("\n--- Training Lasso Regression for Strategy 2 ---")
    result_lasso_s2 = train_evaluate_lasso(
        df=df_strategy2,
        features=features_strategy2,
        strategy_name="Strategy 2: Impute Missing Values",
        alpha=0.1
    )
    strategy_results.append(result_lasso_s2)

    # ---------------------------
    # Ridge Regression
    # ---------------------------
    print("\n--- Training Ridge Regression for Strategy 2 ---")
    result_ridge_s2 = train_evaluate_ridge(
        df=df_strategy2,
        features=features_strategy2,
        strategy_name="Strategy 2: Impute Missing Values",
        alpha=1.0
    )
    strategy_results.append(result_ridge_s2)

    # ---------------------------
    # Custom Gradient Descent
    # ---------------------------
    print("\n--- Training Custom Gradient Descent for Strategy 2 ---")
    result_gd_s2 = train_evaluate_custom_gd(
        df=df_strategy2,
        features=features_strategy2,
        strategy_name="Strategy 2: Impute Missing Values",
        learning_rate=0.001,
        epochs=1000,
        tolerance=1e-6
    )
    strategy_results.append(result_gd_s2)

    # ===============================
    # 4. Select the Best Strategy
    # ===============================
    print("\nSelecting the Best Strategy Based on Test cMSE...")
    best_strategy = select_best_strategy(strategy_results)
    selected_strategy_name = best_strategy['strategy']
    selected_test_cMSE = best_strategy['test_cMSE']

    print(f"\nBest Strategy Selected: {selected_strategy_name} with Test cMSE: {selected_test_cMSE:.4f}")

    # ===============================
    # 5. Retrain the Best Model on Entire Data
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
    if False:  # "Strategy 1: Drop Rows" in selected_strategy_name:
        # Strategy 1: Drop rows with any missing feature values
        kaggle_test_clean = kaggle_test_data.dropna(subset=selected_features)
        print(f"Dropped rows with missing feature values in Kaggle test data: {kaggle_test_clean.shape}")
    else:
        # Calculate the mean for each column in df_original
        column_means = df_original.mean()

        # Create a copy of kaggle_test_data
        kaggle_test_clean = kaggle_test_data.copy()

        # Impute missing values in kaggle_test_clean using the means from df_original
        for col in selected_features:
            if col in kaggle_test_clean.columns:
                mean_value = column_means[col]  # Get mean from df_original
                kaggle_test_clean[col] = kaggle_test_clean[col].fillna(mean_value)
                print(
                    f"Imputed missing values in '{col}' with mean value {mean_value:.2f} from df_original in Kaggle test data")

    # Ensure that the test data has the required features
    missing_features = [feature for feature in selected_features if feature not in kaggle_test_clean.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing in the Kaggle test data: {missing_features}")

    # Extract features for prediction
    X_kaggle_test = kaggle_test_clean[selected_features].values

    # ===============================
    # 6. Retrain the Best Model
    # ===============================
    if "Linear Regression" in selected_strategy_name:
        # Retrain using Scikit-learn Linear Regression
        print("\nRetraining Linear Regression on the Entire Dataset...")
        best_model = LinearRegression()
        best_model.fit(selected_df[selected_features], selected_df[PREDICTION_COLUMN_NAME])
        print("Best Linear Regression model retrained successfully.")
        y_kaggle_pred = best_model.predict(kaggle_test_clean[selected_features])
    elif "Lasso" in selected_strategy_name:
        # Retrain using Lasso Regression
        print("\nRetraining Lasso Regression on the Entire Dataset...")
        alpha_lasso = 0.1
        best_model = Lasso(alpha=alpha_lasso)
        best_model.fit(selected_df[selected_features], selected_df[PREDICTION_COLUMN_NAME])
        print("Best Lasso Regression model retrained successfully.")
        y_kaggle_pred = best_model.predict(kaggle_test_clean[selected_features])
    elif "Ridge" in selected_strategy_name:
        # Retrain using Ridge Regression
        print("\nRetraining Ridge Regression on the Entire Dataset...")
        alpha_ridge = 1.0
        best_model = Ridge(alpha=alpha_ridge)
        best_model.fit(selected_df[selected_features], selected_df[PREDICTION_COLUMN_NAME])
        print("Best Ridge Regression model retrained successfully.")
        y_kaggle_pred = best_model.predict(kaggle_test_clean[selected_features])
    elif "Gradient Descent" in selected_strategy_name:
        # Retrain using Custom Gradient Descent
        print("\nRetraining Custom Gradient Descent on the Entire Dataset...")
        x_full = selected_df[selected_features].values
        y_full = selected_df[PREDICTION_COLUMN_NAME].values
        c_full = selected_df["Censored"].values
        w, b, history = gradient_descent_c_mse(
            x_full,
            y_full,
            c_full,
            learning_rate=0.001,
            epochs=1000,
            tolerance=1e-6
        )
        print("Best Gradient Descent model retrained successfully.")
        y_kaggle_pred = np.dot(X_kaggle_test, w) + b
    else:
        raise ValueError("Unknown model type selected.")

    # ===============================
    # 7. Prepare and Save Submission
    # ===============================
    print("\nPreparing Submission File...")

    # Generate the next submission filename
    submission_filename = get_next_submission_filename(
        base_path="data",
        prefix="cMSE-baseline-submission",
        extension=".csv"
    )
    print(f"Generated Submission Filename: {submission_filename}")

    # Prepare and save the submission
    prepare_submission(kaggle_test_clean, y_kaggle_pred, submission_filename)
    print("Submission process completed successfully.")
