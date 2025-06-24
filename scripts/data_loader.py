import json
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_and_split_data(json_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads data from a JSON file, flattens its structure, and splits it into
    training and validation datasets.

    Args:
        json_path (str): Path to the input JSON file.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed for random splitting for reproducibility.

    Returns:
        tuple: A tuple containing raw_train_dataset and raw_val_dataset (Hugging Face Dataset objects).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                processed_data.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'answers': qa['answers'],
                    'context': paragraph['context']
                })

    # Splitting the processed data into train and test sets
    train_data, val_data = train_test_split(processed_data, test_size=test_size, random_state=random_state)

    # Create Dataset objects for train and test sets
    raw_train_dataset = Dataset.from_dict({
        'id': [item['id'] for item in train_data],
        'question': [item['question'] for item in train_data],
        'answers': [item['answers'] for item in train_data],
        'context': [item['context'] for item in train_data]
    })

    raw_val_dataset = Dataset.from_dict({
        'id': [item['id'] for item in val_data],
        'question': [item['question'] for item in val_data],
        'answers': [item['answers'] for item in val_data],
        'context': [item['context'] for item in val_data]
    })

    print("Train Dataset:")
    print(raw_train_dataset)
    print("\nTest Dataset:")
    print(raw_val_dataset)

    # Optional: Filter out examples where the answer text length is 1 (if desired, based on notebook)
    # raw_train_dataset = raw_train_dataset.filter(lambda x: len(x["answers"][0]["text"]) != 1)
    # raw_val_dataset = raw_val_dataset.filter(lambda x: len(x["answers"][0]["text"]) != 1)

    return raw_train_dataset, raw_val_dataset

if __name__ == '__main__':
    # Example usage (replace with your actual data path)
    # Make sure to create a 'data' directory and place your JSON file there
    # For demonstration, assuming the file is directly in 'data/'
    json_file_path = '../data/BioASQ-train-factoid-4b.json'
    train_ds, val_ds = load_and_split_data(json_file_path)
    print(f"\nTrain dataset size after loading and splitting: {len(train_ds)}")
    print(f"Validation dataset size after loading and splitting: {len(val_ds)}")

