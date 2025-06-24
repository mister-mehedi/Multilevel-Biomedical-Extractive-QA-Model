import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_performance(metrics: dict, train_losses: list, mrr_scores: list, num_train_epochs: int, save_path: str = './performance_graphs'):
    """
    Generates and saves performance plots for training loss, MRR, F1, EM, SACC, and LACC.

    Args:
        metrics (dict): A dictionary containing lists of scores for 'exact_match', 'f1', 'mrr', 'lacc', 'sacc'.
        train_losses (list): A list of training loss values for each epoch.
        mrr_scores (list): A list of MRR scores for each epoch.
        num_train_epochs (int): The total number of training epochs.
        save_path (str): Directory where the plots will be saved.
    """
    epochs = range(1, num_train_epochs + 1)

    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created performance graph directory: {save_path}")

    # Plot Training Loss and MRR
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    plt.plot(epochs, mrr_scores, label='MRR', color='orange', marker='x', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MRR')
    plt.title('MRR Over Epochs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    loss_mrr_filename = os.path.join(save_path, 'training_loss_mrr.png')
    plt.savefig(loss_mrr_filename, dpi=300)
    print(f"Saved: {loss_mrr_filename}")
    plt.close() # Close the figure to free memory

    # Plot Bar charts for F1, EM, SACC, LACC
    metric_names = ['F1', 'EM', 'SACC', 'LACC']
    metrics_scores = [metrics['f1'], metrics['exact_match'], metrics['sacc'], metrics['lacc']]

    plt.figure(figsize=(12, 10))
    for i, (metric_name, scores) in enumerate(zip(metric_names, metrics_scores)):
        plt.subplot(2, 2, i + 1) # 2 rows, 2 columns
        plt.bar(epochs, scores, label=metric_name, color=plt.cm.viridis(i / len(metric_names))) # Use a colormap for varied colors
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Over Epochs')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend()
        plt.ylim(0, 100 if metric_name in ['F1', 'EM'] else 1) # Set y-limit for scores

    plt.tight_layout()
    metrics_over_epochs_filename = os.path.join(save_path, 'metrics_over_epochs.png')
    plt.savefig(metrics_over_epochs_filename, dpi=300)
    print(f"Saved: {metrics_over_epochs_filename}")
    plt.close() # Close the figure

if __name__ == '__main__':
    # Example Usage for visualization
    # Create dummy data for testing
    num_epochs = 10
    dummy_train_losses = np.linspace(2.5, 0.5, num_epochs).tolist()
    dummy_mrr_scores = np.linspace(0.02, 0.05, num_epochs).tolist()
    dummy_em_scores = np.linspace(70, 85, num_epochs).tolist()
    dummy_f1_scores = np.linspace(75, 90, num_epochs).tolist()
    dummy_lacc_scores = np.linspace(0.45, 0.55, num_epochs).tolist()
    dummy_sacc_scores = np.linspace(0.65, 0.80, num_epochs).tolist()

    dummy_metrics = {
        'exact_match': dummy_em_scores,
        'f1': dummy_f1_scores,
        'mrr': dummy_mrr_scores,
        'lacc': dummy_lacc_scores,
        'sacc': dummy_sacc_scores
    }

    print("Generating dummy performance plots...")
    visualize_performance(dummy_metrics, dummy_train_losses, dummy_mrr_scores, num_epochs, save_path='./test_performance_graphs')
    print("Dummy plots generated in 'test_performance_graphs' directory.")

