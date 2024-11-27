import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Constants
DATA_FILE_BASE_PATH = "data/"
FEATURE_COLUMN_NAMES = ["Age", "Gender", "Stage", "GeneticRisk", "TreatmentType", "ComorbidityIndex", "TreatmentResponse"]
TARGET_COLUMN_NAMES = ["SurvivalTime", "Censored"]

# Load and preprocess the dataset
def load_and_preprocess(filename):
    df = pd.read_csv(filename)
    if df.columns[0] != "id":
        df = df.rename(columns={df.columns[0]: "id"})
    return df

# Visualize missing data
def visualize_missing_data(df, title="Missing Data Visualization"):
    print(f"\n{title}")
    if df.empty or df.shape[1] == 0:
        print("DataFrame is empty or has no columns, skipping visualization.")
        return
    plt.figure(figsize=(10, 6))
    msno.bar(df)
    plt.show()

    if df.isnull().sum().sum() > 0:  # Only show these if there are missing values
        plt.figure(figsize=(10, 6))
        msno.matrix(df)
        plt.show()

        plt.figure(figsize=(10, 6))
        msno.heatmap(df)
        plt.show()

        if df.shape[0] > 1:  # Dendrogram requires at least 2 rows
            plt.figure(figsize=(10, 6))
            msno.dendrogram(df)
            plt.show()

# Visualize correlation heatmap
def visualize_correlation(df, title="Correlation Heatmap"):
    print(f"\n{title}")
    if df.empty or df.shape[1] == 0:
        print("DataFrame is empty or has no columns, skipping visualization.")
        return

    correlation_matrix = df.corr()  # Compute correlations for all variables
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", vmin=-1, vmax=1)  # Adjust scale
    plt.title(title)
    plt.show()



def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, test_percentage=0.2, seed=None):
    """
    Split the DataFrame into training, validation, and test sets based on the specified fractions.
    :param df:
    :param train_percent:
    :param validate_percent:
    :param test_percentage:
    :param seed:
    :return:
    """
    assert train_percent + validate_percent + test_percentage == 1.0, "Fractions must sum to 1"

    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test


# Function to build and train the linear regression model
def build_and_train_model(x_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    return model_pipeline


# Main Execution
def plot_y_yhat(y_val, y_val_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, color='blue')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Load dataset
    df = load_and_preprocess(DATA_FILE_BASE_PATH + "train_data.csv")

    # Visualize original data
    visualize_missing_data(df, "Original Data Visualization")

    # Drop rows with missing SurvivalTime
    df = df.dropna(subset=["SurvivalTime"])

    # Drop censored rows (Censored == 1)
    df = df[df["Censored"] == 0]

    # Fill missing values instead of dropping columns
    for col in FEATURE_COLUMN_NAMES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())  # Replace missing values with mean

    # Check remaining data
    print(f"Remaining rows: {len(df)}")
    print(f"Remaining columns: {df.columns.tolist()}")

    # Ensure valid feature columns
    existing_features = [col for col in FEATURE_COLUMN_NAMES if col in df.columns]
    if not existing_features:
        raise ValueError("No valid features found in the DataFrame after cleaning!")

    # Pairplot for valid features and SurvivalTime
    sns.pairplot(df, vars=existing_features + ["SurvivalTime"])
    plt.title("Pairplot for Remaining Features and SurvivalTime")
    plt.show()

    # Define the feature matrix X and target vector y
    X = df[existing_features]
    y = df["SurvivalTime"]
    print("Feature matrix X shape:", X.shape)
    print("Target vector y shape:", y.shape)

    # Visualize full correlation heatmap with all valid data
    visualize_correlation(df, "Full Correlation Heatmap (Valid Data)")

    # Based on correclation heatmap, we can see that the features are not highly correlated with each other
    # With that in mind will drop the Generic Risk and Gender
    existing_features = [col for col in existing_features if col not in ["GenericRisk", "Gender"]]
    print("Remaining features after dropping 'GenericRisk' and 'Gender':", existing_features)

    df_train, df_validation, df_test = train_validate_test_split(df, train_percent=0.6, validate_percent=0.12, test_percentage=0.28, seed=42)
    print(f"Training set: {len(df_train)} samples")
    print(f"Validation set: {len(df_validation)} samples")
    print(f"Test set: {len(df_test)} samples")

    # train the model based on the training set
    x_train = df_train[existing_features]
    y_train = df_train["SurvivalTime"]
    model = build_and_train_model(x_train, y_train)

    # Validate the model based on the validation set
    x_val = df_validation[existing_features]
    y_val = df_validation["SurvivalTime"]
    y_val_pred = model.predict(x_val)

    # Calculate MSE for validation set
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Validation MSE: {val_mse}")

    # Visualize the predicted vs actual values for the validation set
    plot_y_yhat(y_val, y_val_pred, 'Validation Predicted vs Expected')

    # Load test data and prepare it for prediction
    # and drops all rows with missing values in all columns
    test_data = load_and_preprocess(DATA_FILE_BASE_PATH + "test_data.csv")
    test_data = test_data.dropna()
    clean_test_data = test_data[existing_features]

    # Predict on the test data
    y_test_pred = model.predict(clean_test_data)

    # calculate the mse for the test data
    y_test = df_test["SurvivalTime"]
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['0'])
    y_test_pred_df.insert(0, 'id', y_test_pred_df.index)
    y_test_pred_df.to_csv(DATA_FILE_BASE_PATH + 'baseline-submission-01.csv', index=False)
    print("Test predictions saved to test_predictions.csv")
