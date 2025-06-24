import os
import torch
from transformers import AutoTokenizer, default_data_collator

# Import functions from your new scripts
from scripts.data_loader import load_and_split_data
from scripts.preprocessor import preprocess_training_examples, preprocess_validation_examples, CustomDataset, set_dataset_format
from scripts.trainer import train_and_save_model, bootstrap_sampling
from scripts.metrics import compute_metrics # Although primarily used inside trainer, good to import if needed elsewhere
import pickle

def main():
    """
    Orchestrates the entire deep learning pipeline:
    1. Loads and splits data.
    2. Initializes tokenizer.
    3. Preprocesses data for training and validation.
    4. Generates bootstrap samples (if ensemble is desired).
    5. Trains and saves models for each bootstrap sample (or a single model).
    """
    # --- Configuration ---
    json_file_path = './data/BioASQ-train-factoid-4b.json' # Adjust path as needed
    model_checkpoint = "dmis-lab/biobert-v1.1"
    ner_model_path = 'C:/Users/beqa/MEHEDI/Notebook/ner all dataset/biobert_model_ner' # Local path to your NER model
    save_directory = "./checkpoints"
    num_train_epochs = 10
    n_bootstrap_samples = 1 # Set to 1 for single model training, >1 for bagging
    learning_rate = 2e-5
    n_best = 20
    max_answer_length = 30

    # Ensure save directory exists
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.join(save_directory, 'performance_graphs'), exist_ok=True)

    print("--- Step 1: Loading and Splitting Data ---")
    raw_train_dataset, raw_val_dataset = load_and_split_data(json_file_path)

    # Initialize tokenizer for preprocessing
    print("\n--- Step 2: Initializing Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("\n--- Step 3: Preprocessing Data ---")
    # Preprocess training data
    train_dataset = raw_train_dataset.map(
        lambda examples: preprocess_training_examples(examples, tokenizer),
        batched=True,
        remove_columns=raw_train_dataset.column_names,
    )
    print(f"Processed training dataset size: {len(train_dataset)}")

    # Preprocess validation data
    validation_dataset = raw_val_dataset.map(
        lambda examples: preprocess_validation_examples(examples, tokenizer),
        batched=True,
        remove_columns=raw_val_dataset.column_names,
    )
    print(f"Processed validation dataset size: {len(validation_dataset)}")

    print("\n--- Step 4: Generating Bootstrap Samples (if n_bootstrap_samples > 1) ---")
    if n_bootstrap_samples > 1:
        bootstrap_samples = bootstrap_sampling(train_dataset, n_bootstrap_samples)
        # Optionally, save bootstrap samples to disk
        bootstrap_samples_path = os.path.join(save_directory, 'bootstrap_samples.pkl')
        with open(bootstrap_samples_path, 'wb') as f:
            pickle.dump(bootstrap_samples, f)
        print(f"Bootstrap samples saved to '{bootstrap_samples_path}'")
    else:
        # If not bagging, just use the full preprocessed train_dataset as the "single sample"
        bootstrap_samples = [train_dataset]
        print("Training a single model (no bootstrap sampling).")


    # --- Step 5: Training Models ---
    print("\n--- Step 5: Training Models ---")
    for i, current_train_sample in enumerate(bootstrap_samples):
        print(f"\nTraining model {i+1}/{len(bootstrap_samples)}...")
        train_and_save_model(
            train_dataset=current_train_sample,
            validation_dataset=validation_dataset,
            raw_val_dataset=raw_val_dataset, # Pass original for metric computation
            model_checkpoint=model_checkpoint,
            save_dir=save_directory,
            model_idx=i,
            num_train_epochs=num_train_epochs,
            n_best=n_best,
            max_answer_length=max_answer_length,
            learning_rate=learning_rate,
            ner_model_path=ner_model_path
        )
    
    print("\n--- Training pipeline completed! ---")
    print(f"Models and performance graphs saved in the '{save_directory}' directory.")

if __name__ == '__main__':
    main()

