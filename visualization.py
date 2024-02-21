
import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, dataset_type='Training'):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.3,
                label=f'{dataset_type} data')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(),
             y_true.max()], 'k--', lw=4, label='Ideal')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{dataset_type} Data: Actual vs Predicted Prices')
    plt.legend()
    plt.show()
