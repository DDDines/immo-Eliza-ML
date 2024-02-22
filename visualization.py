
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_actual, y_predicted, dataset_type="Training"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.3)
    plt.title(f"{dataset_type} Dataset: Actual vs. Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.plot([y_actual.min(), y_actual.max()], [
             y_actual.min(), y_actual.max()], 'k--', lw=4)
    plt.show()


def plot_residuals_histogram(y_actual, y_predicted, dataset_type="Training"):
    residuals = y_actual - y_predicted
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title(f"{dataset_type} Dataset: Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()
