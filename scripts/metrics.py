import numpy as np
import collections
import evaluate

# Load the SQuAD metric from Hugging Face evaluate library
metric = evaluate.load("squad")

def compute_lacc(predictions, references):
    """
    Computes Lacc (Local Accuracy) for question answering predictions.
    Lacc measures the accuracy of predicted start and end token positions.

    Args:
        predictions (list): A list of dictionaries, where each dictionary
                            contains 'prediction_start' and 'prediction_end' keys.
        references (list): A list of dictionaries, where each dictionary
                           contains 'answers' (a list of answer dictionaries)
                           with 'answer_start' and 'text' keys.

    Returns:
        float: The Lacc score.
    """
    correct_start = 0
    correct_end = 0
    total = 0

    for pred, ref in zip(predictions, references):
        # Ensure there's at least one answer in references
        if ref['answers'] and ref['answers'][0]['text']:
            pred_start = pred['prediction_start']
            pred_end = pred['prediction_end']
            ref_start = ref['answers'][0]['answer_start']
            ref_end = ref['answers'][0]['answer_start'] + len(ref['answers'][0]['text'])

            if pred_start == ref_start:
                correct_start += 1
            if pred_end == ref_end:
                correct_end += 1
            total += 1
    
    if total == 0:
        return 0.0 # Avoid division by zero
    
    lacc = (correct_start + correct_end) / (2 * total)
    return lacc

def compute_sacc(predictions, references):
    """
    Computes Sacc (Sequence Accuracy) for question answering predictions.
    Sacc measures the exact match accuracy of the predicted answer text.

    Args:
        predictions (list): A list of dictionaries, where each dictionary
                            contains a 'prediction_text' key.
        references (list): A list of dictionaries, where each dictionary
                           contains 'answers' (a list of answer dictionaries)
                           with a 'text' key.

    Returns:
        float: The Sacc score.
    """
    correct_sequences = 0
    total = 0

    for pred, ref in zip(predictions, references):
        # Ensure there's at least one answer in references
        if ref['answers'] and ref['answers'][0]['text']:
            pred_text = pred['prediction_text']
            ref_text = ref['answers'][0]['text']

            if pred_text == ref_text:
                correct_sequences += 1
            total += 1
    
    if total == 0:
        return 0.0 # Avoid division by zero

    sacc = correct_sequences / total
    return sacc

def compute_mrr(predictions, references):
    """
    Computes Mean Reciprocal Rank (MRR) for question answering predictions.
    MRR is the average of the reciprocal ranks of the first correct answer.

    Args:
        predictions (list): A list of dictionaries, where each dictionary
                            contains 'prediction_text' (can be a single string or list of strings if multiple predictions are ranked).
        references (list): A list of dictionaries, where each dictionary
                           contains 'answers' (a list of answer dictionaries)
                           with a 'text' key.

    Returns:
        float: The MRR score.
    """
    reciprocal_ranks = []
    for pred, ref in zip(predictions, references):
        if ref["answers"]:
            correct_answers = [a["text"] for a in ref["answers"] if a["text"]]
            if not correct_answers: # Skip if no valid correct answers
                reciprocal_ranks.append(0.0)
                continue

            # Assuming pred["prediction_text"] is the single best predicted text for SQuAD-like format
            # If `pred["prediction_text"]` can be a list of ranked predictions, adjust this logic.
            # For a single prediction:
            if pred["prediction_text"] in correct_answers:
                reciprocal_ranks.append(1.0) # Rank 1
            else:
                reciprocal_ranks.append(0.0) # Not found

    if not reciprocal_ranks:
        return 0.0 # Avoid division by zero

    return np.mean(reciprocal_ranks)


def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
    """
    Computes a full set of metrics (F1, EM, MRR, Lacc, Sacc) for Question Answering.

    Args:
        start_logits (np.ndarray): Predicted start logits for each feature.
        end_logits (np.ndarray): Predicted end logits for each feature.
        features (Dataset): Processed features dataset (e.g., validation_dataset)
                            containing 'example_id', 'offset_mapping'.
        examples (Dataset): Original examples dataset (e.g., raw_val_dataset)
                            containing 'id', 'question', 'answers', 'context'.
        n_best (int): The number of best start/end logits to consider.
        max_answer_length (int): The maximum allowed length for a predicted answer.

    Returns:
        dict: A dictionary containing the computed metrics (exact_match, f1, mrr, lacc, sacc).
    """
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            # Use argsort to get indices of top n_best logits
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        "prediction_start": offsets[start_index][0],
                        "prediction_end": offsets[end_index][1],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"],
                 "prediction_start": best_answer["prediction_start"],
                 "prediction_end": best_answer["prediction_end"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "",
                                      "prediction_start": -1, "prediction_end": -1})

    # Prepare theoretical answers for metric computation
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    # Compute F1 and Exact Match using the SQuAD metric
    squad_predictions_format = [{"id": p["id"], "prediction_text": p["prediction_text"]} for p in predicted_answers]
    squad_references_format = theoretical_answers
    
    # Ensure references are in the correct format expected by evaluate.load("squad")
    # It expects: references=[{'id': 'xyz', 'answers': {'answer_start': [123], 'text': ['abc']}}]
    # Our theoretical_answers already have 'id' and 'answers' which is a list of dicts with 'text' and 'answer_start'.
    # The `evaluate` library usually handles this well.
    metrics = metric.compute(predictions=squad_predictions_format, references=squad_references_format)

    # Calculate Lacc and Sacc
    metrics['mrr'] = compute_mrr(predicted_answers, theoretical_answers)
    metrics['lacc'] = compute_lacc(predicted_answers, theoretical_answers)
    metrics['sacc'] = compute_sacc(predicted_answers, theoretical_answers)

    return metrics

def compute_statistics(scores):
    """
    Computes highest, lowest, and average from a list of scores.

    Args:
        scores (list): A list of numerical scores.

    Returns:
        dict: A dictionary containing 'highest', 'lowest', and 'average' scores.
    """
    if not scores:
        return {'highest': None, 'lowest': None, 'average': None}
    return {
        'highest': np.max(scores),
        'lowest': np.min(scores),
        'average': np.mean(scores)
    }

if __name__ == '__main__':
    # This block is for demonstration and testing the metrics.
    # In a real training pipeline, these functions would be called by trainer.py

    # Dummy data for demonstration
    dummy_features = [
        {"example_id": "id1", "offset_mapping": [[0,5], [6,11], [12,17], None]},
        {"example_id": "id1", "offset_mapping": [[0,5], [6,11], [12,17], None]},
        {"example_id": "id2", "offset_mapping": [[0,4], [5,10], [11,16]]}
    ]
    dummy_examples = [
        {"id": "id1", "question": "What is x?", "answers": [{"text": "hello", "answer_start": 0}], "context": "hello world example"},
        {"id": "id2", "question": "What is y?", "answers": [{"text": "test", "answer_start": 5}], "context": "a test string"}
    ]
    dummy_start_logits = np.array([
        [1.0, 0.5, 0.2, 0.1], # For id1, feature 0
        [0.1, 0.2, 1.0, 0.5], # For id1, feature 1 (different offsets, but same example_id)
        [0.3, 1.0, 0.6]       # For id2, feature 0
    ])
    dummy_end_logits = np.array([
        [0.8, 0.9, 0.3, 0.1], # For id1, feature 0
        [0.2, 0.3, 0.8, 1.0], # For id1, feature 1
        [0.7, 0.9, 1.0]       # For id2, feature 0
    ])

    # Convert to Hugging Face Dataset format for `features` to simulate actual usage
    from datasets import Dataset
    features_ds = Dataset.from_list(dummy_features)
    examples_ds = Dataset.from_list(dummy_examples)

    print("--- Testing compute_metrics ---")
    metrics_result = compute_metrics(dummy_start_logits, dummy_end_logits, features_ds, examples_ds, n_best=2, max_answer_length=10)
    print("Computed Metrics:", metrics_result)

    # Test compute_statistics
    print("\n--- Testing compute_statistics ---")
    em_scores = [78.89, 86.08, 83.02, 80.88, 84.40]
    f1_scores = [80.83, 88.21, 84.85, 83.13, 86.42]
    mrr_scores = [0.032, 0.033, 0.032, 0.033, 0.032]
    lacc_scores = [0.506, 0.491, 0.479, 0.469, 0.475]
    sacc_scores = [0.732, 0.801, 0.743, 0.759, 0.781]

    metrics_statistics = {
        'EM': compute_statistics(em_scores),
        'F1': compute_statistics(f1_scores),
        'MRR': compute_statistics(mrr_scores),
        'Lacc': compute_statistics(lacc_scores),
        'Sacc': compute_statistics(sacc_scores),
    }

    for metric_name, stats in metrics_statistics.items():
        print(f"{metric_name} - Highest: {stats['highest']}, Lowest: {stats['lowest']}, Average: {stats['average']:.4f}")

