import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import missingno as msno

DATA_FILE_BASE_PATH = "data/"

TRAIN_COLUMN_NAMES = ["id", "Age", "Gender", "Stage", "GeneticRisk", "TreatmentType", "ComorbidityIndex",
                      "TreatmentResponse", "SurvivalTime", "Censored"]

TARGET_COLUMN_NAMES = ["SurvivalTime", "Censored"]
FEATURE_COLUMN_NAMES = ["Age", "Gender", "Stage", "GeneticRisk", "TreatmentType", "ComorbidityIndex", "TreatmentResponse"]


# Load and preprocess the dataset
def load_and_preprocess(filename):
    df = pd.read_csv(filename)

    # Correct the nameless 1st column to Id if necessary
    if df.columns[0] != "id":
        df = df.rename(columns={df.columns[0]: "id"})

    # # Drop rows with NaN values
    # df = df.dropna()

    # Split into features (X) and target (y)
    # df_x = df[FEATURE_COLUMN_NAMES]
    # df_y = df[TARGET_COLUMN_NAMES]

    # return df_x, df_y
    return df

# Function to build and train the model
def build_and_train_model(x_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    return model_pipeline


def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


# Main execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    # df_x, df_y = load_and_preprocess(DATA_FILE_BASE_PATH + "train_data.csv")
    df = load_and_preprocess(DATA_FILE_BASE_PATH + "train_data.csv")
    msno.bar(df)

    # # Split the data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=42)
    #
    # # Build and train the model
    # model = build_and_train_model(x_train, y_train)
    #
    # # Evaluate the model on the training set
    # y_train_pred = model.predict(x_train)
    # train_mse = mean_squared_error(y_train, y_train_pred)
    # print(f"Training MSE: {train_mse}")
    #
    # # Evaluate the model on the testing set
    # y_test_pred = model.predict(x_test)
    # test_mse = mean_squared_error(y_test, y_test_pred)
    # print(f"Testing MSE: {test_mse}")
    #
    # # Save predictions
    # y_test_pred_df = pd.DataFrame(y_test_pred, columns=TARGET_COLUMN_NAMES)
    # y_test_pred_df.insert(0, 'Id', x_test.index)  # Reinsert the Id column
    # y_test_pred_df.to_csv(DATA_FILE_BASE_PATH + 'before_baseline.csv', index=False)
    # print("Predictions saved to 'before_baseline.csv'")
