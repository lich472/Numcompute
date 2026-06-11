import matplotlib.pyplot as plt
import numpy as np

def plot_metric_over_time(metric_values, title, ylabel, save_path=None):
    
    plt.figure() # create figure in advance
    
    chunks = list(range(1, len(metric_values) + 1)) # x axis starts from 1
    
    plt.plot(chunks, metric_values, marker='o', color='blue', linestyle='-')
    
    plt.title(title)
    plt.xlabel('Chunk Number')
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # options for saving to file or inline display
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_models(metric1, metric2, labels, title='Model Comparison', save_path=None):
    plt.figure()
    
    chunks = list(range(1, len(metric1) + 1))
    
    # two lines
    plt.plot(chunks, metric1, marker='s', label=labels[0], color='green')
    plt.plot(chunks, metric2, marker='^', label=labels[1], color='red')
    
    plt.title(title)
    plt.xlabel('Chunk Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None):
    plt.figure()
    
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    indices = np.arange(len(y_true_arr))
    
    plt.scatter(indices, y_true_arr, color='black', label='Actual', marker='o', s=100, alpha=0.6)
    plt.scatter(indices, y_pred_arr, color='orange', label='Predicted', marker='x', s=50, alpha=0.8)
    
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Sample Index (Current Chunk)')
    plt.ylabel('Class Label / Value')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()