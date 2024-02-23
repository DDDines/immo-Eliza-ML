import pandas as pd
from train import train
import polynominal
from visualization import plot_actual_vs_predicted
from predict_ import predict

data = pd.read_csv("data/properties.csv")

data.head()


def outliers(data, column, lower_bound_multiplier=1.5, upper_bound_multiplier=1.5, return_outliers=False):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Definir limites para identificar outliers
    lower_bound = Q1 - lower_bound_multiplier * IQR
    upper_bound = Q3 + upper_bound_multiplier * IQR

    # Identificar outliers
    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    outliers_data = data[outliers_mask]

    # Remover outliers
    data_cleaned = data[~outliers_mask]

    if return_outliers:
        return data_cleaned, outliers_data
    else:
        return data_cleaned


if __name__ == "__main__":
    # Iterar apenas por colunas numéricas
    numeric_cols = data.select_dtypes(include=['number']).columns
    # data = clean_and_standardize_data(data)

    for column in numeric_cols:
        # Atualizar 'data' após remover outliers de cada coluna
        data = outliers(data, column)

    y_train, y_train_pred, y_test, y_test_pred = polynominal.train(data)

    # Plotting for Training Dataset
    # plot_actual_vs_predicted(y_train, y_train_pred, "Training")
    # plot_residuals_histogram(y_train, y_train_pred, "Training")

    # Plotting for Testing Dataset
    # plot_actual_vs_predicted(y_test, y_test_pred, "Testing")
    # plot_residuals_histogram(y_test, y_test_pred, "Testing")

    # predict()

    # pred = pd.read_csv("output\predictions.csv")
   # property = pd.read_csv("data/properties.csv")

   # plot_actual_vs_predicted(pred["predictions"], property["price"])
