from matplotlib import pyplot as plt
import missingno as msno

from model import load_and_preprocess, DATA_FILE_BASE_PATH

def visualize_data(df):
    # Visualize missing data
    plt.figure(figsize=(10, 6))
    msno.bar(df)
    plt.show()

    plt.figure(figsize=(10, 6))
    msno.matrix(df)
    plt.show()

    plt.figure(figsize=(10, 6))
    msno.heatmap(df)
    plt.show()

    plt.figure(figsize=(10, 6))
    msno.dendrogram(df)
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = load_and_preprocess(DATA_FILE_BASE_PATH + "train_data.csv")

    # Visualize missing data
    visualize_data(df)
    df_temp = df.dropna()
    visualize_data(df_temp)

    # droping only values in the SurvivalTime column
    df_temp = df.dropna(subset=["SurvivalTime"])

