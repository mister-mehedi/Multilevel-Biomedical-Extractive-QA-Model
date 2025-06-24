from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import collections

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle lists of dictionaries,
    converting dictionary values to PyTorch tensors for batching.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Handle batching for DataLoader
        if isinstance(idx, list):
            batch = {key: [] for key in self.data_list[0].keys()}
            for i in idx:
                for key, value in self.data_list[i].items():
                    batch[key].append(value)
            # Stack tensors for keys that are lists of tensors, others remain lists
            return {key: torch.stack(value) if isinstance(value[0], torch.Tensor) else value for key, value in batch.items()}
        
        # Ensure individual items are tensors if applicable
        item = self.data_list[idx]
        return {key: torch.tensor(value) if isinstance(value, (list, int, float)) else value for key, value in item.items()}


def preprocess_training_examples(examples, tokenizer, max_length=512, stride=256):
    """
    Preprocesses training examples for a Question Answering task.
    This function tokenizes questions and contexts, handles overflowing tokens,
    and computes start and end positions of answers.

    Args:
        examples (dict): A dictionary of examples containing 'question', 'context', and 'answers'.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        max_length (int): The maximum total input sequence length after tokenization.
        stride (int): The stride (overlap) between chunks when `truncation="only_second"`
                      and `return_overflowing_tokens=True`.

    Returns:
        BatchEncoding: A Hugging Face BatchEncoding object with processed inputs,
                       including 'start_positions' and 'end_positions'.
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx][0] # Assuming only one answer per example for simplicity
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples, tokenizer, max_length=512, stride=256):
    """
    Preprocesses validation examples for a Question Answering task.
    This function tokenizes questions and contexts, handles overflowing tokens,
    and prepares `offset_mapping` and `example_id` for metric computation.

    Args:
        examples (dict): A dictionary of examples containing 'question' and 'context'.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        max_length (int): The maximum total input sequence length after tokenization.
        stride (int): The stride (overlap) between chunks when `truncation="only_second"`
                      and `return_overflowing_tokens=True`.

    Returns:
        BatchEncoding: A Hugging Face BatchEncoding object with processed inputs,
                       including 'example_id' and updated 'offset_mapping'.
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def set_dataset_format(dataset, format_type):
    """
    Sets the format of the dataset. For 'torch', converts relevant columns to tensors.
    """
    if format_type == "torch":
        # This will convert columns like 'input_ids', 'attention_mask', 'token_type_ids',
        # 'start_positions', 'end_positions' to torch tensors if they are lists/arrays.
        # 'offset_mapping' and 'example_id' are typically not converted to tensors.
        for feature in dataset.features:
            if feature not in ['offset_mapping', 'example_id']: # These usually remain as lists/ints
                dataset.set_format(type=format_type, columns=[feature])
    return dataset


if __name__ == '__main__':
    from data_loader import load_and_split_data
    from transformers import default_data_collator

    # Initialize tokenizer (replace with your desired model checkpoint)
    model_checkpoint = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load raw data
    json_file_path = '../data/BioASQ-train-factoid-4b.json'
    raw_train_dataset, raw_val_dataset = load_and_split_data(json_file_path)

    # Preprocess training data
    train_dataset = raw_train_dataset.map(
        lambda examples: preprocess_training_examples(examples, tokenizer),
        batched=True,
        remove_columns=raw_train_dataset.column_names,
    )
    print(f"\nOriginal train dataset length: {len(raw_train_dataset)}")
    print(f"Processed train dataset length: {len(train_dataset)}")
    print("Processed Train Dataset features:", train_dataset.features)


    # Preprocess validation data
    validation_dataset = raw_val_dataset.map(
        lambda examples: preprocess_validation_examples(examples, tokenizer),
        batched=True,
        remove_columns=raw_val_dataset.column_names,
    )
    print(f"\nOriginal validation dataset length: {len(raw_val_dataset)}")
    print(f"Processed validation dataset length: {len(validation_dataset)}")
    print("Processed Validation Dataset features:", validation_dataset.features)

    # Set format to torch and wrap with CustomDataset
    train_dataset = set_dataset_format(train_dataset, "torch")
    train_dataset_custom = CustomDataset(train_dataset)

    # Create a small DataLoader to test
    # Note: `default_data_collator` from transformers automatically handles padding
    # when using `return_tensors="pt"` during tokenization, which is what `set_format("torch")` does.
    # The `CustomDataset` is primarily for iterating through the dataset with correct tensor types.
    # The `default_data_collator` is still needed for handling batching of the dictionary outputs.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_custom,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8,
    )

    print("\nTesting DataLoader for training set:")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, Dtype: {v.dtype}")
            else:
                print(f"  {k}: {type(v)}, Length: {len(v)}")
        if i == 0: # Print only the first batch
            break

