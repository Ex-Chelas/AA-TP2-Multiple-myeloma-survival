import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from model import (
    load_and_preprocess, prepare_submission,
    select_best_strategy, strategy_impute_missing, strategy_drop_rows, TRAIN_FILE_PATH,
    train_validate_split, PREDICTION_COLUMN_NAME, TEST_FILE_PATH, get_next_submission_filename,
    error_metric
)

def build_and_train_poly_model(x_train, y_train, degree, regressor):
    """
    Build and train a polynomial regression model using a pipeline with standardization.

    Parameters:
    x_train: The feature matrix for training.
    y_train: The target vector for training.
    degree: The degree of the polynomial features.
    regressor: The regression model (e.g., Ridge, Lasso).

    Returns:
    The trained model pipeline and the number of features generated.
    """
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),  # Polynomial transformation
        ('regressor', regressor)  # Regression model
    ])
    model_pipeline.fit(x_train, y_train)
    num_features = model_pipeline.named_steps['poly_features'].n_output_features_
    print(f"Polynomial Regression model trained successfully with degree {degree} and {num_features} features.")
    return model_pipeline, num_features

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
    # Polynomial Model (Strategy 2)
    # ---------------------------------
    print("\n=== Training Polynomial Regression with Strategy 2 ===")

    # Split data
    df_train_s2, df_validate_s2, df_test_s2 = train_validate_split(df_strategy2)

    # Prepare datasets
    x_train = df_train_s2[features_strategy2]
    y_train = df_train_s2[PREDICTION_COLUMN_NAME]
    x_val = df_validate_s2[features_strategy2]
    y_val = df_validate_s2[PREDICTION_COLUMN_NAME]
    c_val = df_validate_s2["Censored"].values

    # Validate polynomial regression with degrees from 1 to 5
    best_rmse = float('inf')
    best_model = None
    best_degree = None
    for degree in range(1, 6):
        print(f"Validating polynomial regression with degree {degree}...")
        regressor = RidgeCV(alphas=(0.1, 1.0, 10.0))
        poly_model, num_features = build_and_train_poly_model(x_train, y_train, degree, regressor)

        # Predict on validation set
        y_val_pred = poly_model.predict(x_val)
        val_rmse = np.sqrt(error_metric(y_val, y_val_pred, c_val))
        print(f"Degree {degree} validation RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = poly_model
            best_degree = degree

    print(f"\nBest polynomial regression model: Degree {best_degree} with RMSE {best_rmse:.4f}")
    strategy_results.append({
        'strategy': f"Polynomial Regression - Degree {best_degree}",
        'validation_cMSE': best_rmse,
        'test_cMSE': None  # Test RMSE will be computed later
    })

    # ===============================
    # 4. Select the Best Strategy
    # ===============================
    print("\nSelecting the Best Strategy Based on Test cMSE...")
    #best_strategy = select_best_strategy(strategy_results)
    #selected_strategy_name = best_strategy['strategy']
    #selected_test_cMSE = best_strategy['test_cMSE']

    #print(f"\nBest Strategy Selected: {selected_strategy_name} with Test cMSE: {selected_test_cMSE}")

    # ===============================
    # 5. Predict and Save Submission
    # ===============================
    print("\nPreparing Submission File...")

    # Load and preprocess Kaggle test data
    kaggle_test_data = load_and_preprocess(TEST_FILE_PATH)
    print("Kaggle Test DataFrame shape:", kaggle_test_data.shape)

    if False:#"Strategy 1: Drop Rows" in selected_strategy_name:
        # Strategy 1: Drop rows with any missing feature values
        kaggle_test_clean = kaggle_test_data.dropna(subset=selected_features)
        print(f"Dropped rows with missing feature values in Kaggle test data: {kaggle_test_clean.shape}")
    else:
        # Calculate the mean for each column in df_original
        column_means = df_original.mean()

        # Create a copy of kaggle_test_data
        kaggle_test_clean = kaggle_test_data.copy()

        # Impute missing values in kaggle_test_clean using the means from df_original
        for col in df_strategy2:
            if col in kaggle_test_clean.columns:
                mean_value = column_means[col]  # Get mean from df_original
                kaggle_test_clean[col] = kaggle_test_clean[col].fillna(mean_value)
                print(
                    f"Imputed missing values in '{col}' with mean value {mean_value:.2f} from df_original in Kaggle test data")

    # Apply polynomial feature transformation on Kaggle test data
    x_kaggle_test = kaggle_test_clean[features_strategy2]
    y_kaggle_pred = best_model.predict(x_kaggle_test)

    # Generate the next submission filename
    submission_filename = get_next_submission_filename(
        base_path="data",
        prefix="polySubmission",
        extension=".csv"
    )
    print(f"Generated Submission Filename: {submission_filename}")

    # Save predictions
    prepare_submission(kaggle_test_data, y_kaggle_pred, submission_filename)
    print("Submission process completed successfully.")
