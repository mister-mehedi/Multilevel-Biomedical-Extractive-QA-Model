import os
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AdamW, get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import random
import numpy as np
import pickle # For loading/saving bootstrap samples

# Import functions from other scripts
from .model import initialize_model
from .metrics import compute_metrics, compute_statistics
from .visualizer import visualize_performance
from .preprocessor import CustomDataset, set_dataset_format

def bootstrap_sampling(data, n_samples: int):
    """
    Generates bootstrap samples from a given dataset.

    Args:
        data (Dataset): The dataset to sample from.
        n_samples (int): The number of bootstrap samples to generate.

    Returns:
        list: A list of datasets, where each dataset is a bootstrap sample.
    """
    print(f"Generating {n_samples} bootstrap samples...")
    # datasets.Dataset does not have len() or random.choices directly
    # Convert to a list for sampling if it's not already, or use indices
    data_indices = list(range(len(data)))
    bootstrap_samples_indices = [random.choices(data_indices, k=len(data_indices)) for _ in range(n_samples)]
    
    # Create new Dataset objects from the sampled indices
    # This creates a list of lists of dictionaries for each sample
    bootstrap_samples_list = []
    for sample_indices in bootstrap_samples_indices:
        # Select rows using .select() for efficient subsetting
        bootstrap_samples_list.append(data.select(sample_indices))
    
    print("Bootstrap samples generated.")
    return bootstrap_samples_list


def train_and_save_model(
    train_dataset,
    validation_dataset,
    raw_val_dataset, # Original validation dataset needed for compute_metrics
    model_checkpoint: str,
    save_dir: str,
    model_idx: int,
    num_train_epochs: int,
    n_best: int = 20,
    max_answer_length: int = 30,
    learning_rate: float = 2e-5,
    ner_model_path: str = None # Path to NER model for weights transfer
):
    """
    Trains a Question Answering model, evaluates it, and saves the trained model.

    Args:
        train_dataset (Dataset): The preprocessed training dataset (Hugging Face Dataset).
        validation_dataset (Dataset): The preprocessed validation dataset (Hugging Face Dataset).
        raw_val_dataset (Dataset): The original, unprocessed validation dataset (Hugging Face Dataset)
                                   needed for metric computation.
        model_checkpoint (str): Hugging Face model checkpoint name (e.g., "dmis-lab/biobert-v1.1").
        save_dir (str): Directory to save the trained model and tokenizer.
        model_idx (int): Index of the current model in case of multiple training runs (e.g., for bagging).
        num_train_epochs (int): Number of training epochs.
        n_best (int): Number of best start/end logits to consider for answer extraction.
        max_answer_length (int): Maximum length of extracted answer.
        learning_rate (float): Learning rate for the optimizer.
        ner_model_path (str, optional): Path to a pre-trained NER model to initialize BERT encoder.
    """
    print(f"\n--- Starting training for model {model_idx} ---")

    # 1. Model Initialization
    model = initialize_model(model_checkpoint, ner_model_path=ner_model_path)

    # Initialize tokenizer for saving (if it's the first model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Set device (CPU/GPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Model {model_idx} moved to: {device}")

    # 2. Data Preparation for PyTorch DataLoader
    # Apply set_dataset_format to convert relevant columns to torch tensors
    train_dataset_formatted = set_dataset_format(train_dataset, "torch")
    validation_dataset_formatted = set_dataset_format(validation_dataset, "torch")

    # Wrap with CustomDataset to handle list of dictionaries correctly if needed
    # Note: `datasets` library's `set_format("torch")` combined with `default_data_collator`
    # is usually sufficient for standard QA tasks. CustomDataset might be an overhead
    # unless you have very specific data structures. Keeping it for now as it was in notebook.
    train_dataset_custom = CustomDataset(train_dataset_formatted)
    # The validation_set for eval_dataloader needs example_id and offset_mapping removed
    # These columns are handled separately by compute_metrics
    validation_set_for_dataloader = validation_dataset_formatted.remove_columns(["example_id", "offset_mapping"])


    train_dataloader = DataLoader(
        train_dataset_custom, # Use CustomDataset instance
        shuffle=True,
        collate_fn=default_data_collator, # Use default_data_collator for batching
        batch_size=8,
    )

    eval_dataloader = DataLoader(
        validation_set_for_dataloader,
        collate_fn=default_data_collator,
        batch_size=8,
    )

    # 3. Optimizer and Scheduler Setup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator() # For distributed training and mixed precision

    # Prepare model, optimizer, and dataloaders with Accelerate
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Lists to store metrics for each epoch
    train_losses = []
    em_scores = []
    f1_scores = []
    mrr_scores = []
    lacc_scores = []
    sacc_scores = []

    progress_bar = tqdm(range(num_training_steps), desc=f"Training Model {model_idx}")

    # 4. Training Loop
    for epoch in range(num_train_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)

        # Evaluation phase
        model.eval()
        start_logits = []
        end_logits = []
        accelerator.print(f"Model {model_idx}: Evaluation for Epoch {epoch+1}!")
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        
        # Trim logits to match the actual number of validation features,
        # in case of padding or distributed training artifacts
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        # Compute all metrics
        metrics = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_val_dataset, n_best, max_answer_length
        )

        em_scores.append(metrics['exact_match'])
        f1_scores.append(metrics['f1'])
        mrr_scores.append(metrics['mrr'])
        lacc_scores.append(metrics['lacc'])
        sacc_scores.append(metrics['sacc'])

        accelerator.print(f"Epoch {epoch+1} Metrics for Model {model_idx}:")
        accelerator.print(f"Training Loss: {average_train_loss:.4f}")
        accelerator.print(f"EM: {metrics['exact_match']:.2f}")
        accelerator.print(f"F1: {metrics['f1']:.2f}")
        accelerator.print(f"MRR: {metrics['mrr']:.4f}")
        accelerator.print(f"Lacc: {metrics['lacc']:.4f}")
        accelerator.print(f"Sacc: {metrics['sacc']:.4f}")

    # Print overall statistics after training all epochs
    metrics_statistics = {
        'EM': compute_statistics(em_scores),
        'F1': compute_statistics(f1_scores),
        'MRR': compute_statistics(mrr_scores),
        'Lacc': compute_statistics(lacc_scores),
        'Sacc': compute_statistics(sacc_scores),
    }

    accelerator.print(f"\n--- Overall Metrics Statistics for Model {model_idx} ---")
    for metric_name, stats in metrics_statistics.items():
        accelerator.print(f"{metric_name} - Highest: {stats['highest']:.4f}, Lowest: {stats['lowest']:.4f}, Average: {stats['average']:.4f}")

    # Visualize performance
    visualize_performance(
        {'exact_match': em_scores, 'f1': f1_scores, 'mrr': mrr_scores, 'lacc': lacc_scores, 'sacc': sacc_scores},
        train_losses, mrr_scores, num_train_epochs, save_path=os.path.join(save_dir, 'performance_graphs')
    )

    # Save the trained model
    model_save_path = os.path.join(save_dir, f"model_{model_idx}")
    accelerator.wait_for_everyone() # Ensure all processes complete before saving
    unwrapped_model = accelerator.unwrap_model(model) # Get the base model if using DDP/FP16
    unwrapped_model.save_pretrained(model_save_path)
    
    # Save the tokenizer only once, typically with the first model or in a dedicated setup script
    if model_idx == 0:
        tokenizer.save_pretrained(save_dir)
    
    print(f"Model {model_idx} saved to: {model_save_path}")

