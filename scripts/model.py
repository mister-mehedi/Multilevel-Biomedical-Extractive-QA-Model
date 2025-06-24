from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering, AutoConfig
import torch

def initialize_model(model_checkpoint: str, ner_model_path: str = None):
    """
    Initializes a Question Answering model from a given checkpoint.
    Optionally, integrates weights from a pre-trained NER model if provided.

    Args:
        model_checkpoint (str): Hugging Face model checkpoint name (e.g., "dmis-lab/biobert-v1.1").
        ner_model_path (str, optional): Local path to a pre-trained NER model (BertForQuestionAnswering).
                                        If provided, its BERT encoder weights will be loaded into the QA model.
                                        Defaults to None.

    Returns:
        AutoModelForQuestionAnswering: The initialized Question Answering model.
    """
    print(f"Loading QA model from checkpoint: {model_checkpoint}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    if ner_model_path:
        print(f"Loading NER model from path: {ner_model_path}")
        try:
            # Load the NER model configuration and then the model
            ner_config = AutoConfig.from_pretrained(ner_model_path)
            ner_model = BertForQuestionAnswering.from_pretrained(ner_model_path, config=ner_config)
            
            # Transfer the BERT encoder weights
            # The QA model's 'bert' attribute should match the NER model's 'bert' attribute
            model.bert = ner_model.bert
            print("Successfully loaded BERT encoder weights from NER model into QA model.")
        except Exception as e:
            print(f"Warning: Could not load NER model from {ner_model_path}. Error: {e}")
            print("Proceeding with only the base QA model weights.")

    return model

if __name__ == '__main__':
    # Example Usage
    model_checkpoint_name = "dmis-lab/biobert-v1.1"
    # Replace with an actual path to a NER model if you have one
    # For demonstration, we'll try to load a non-existent path first to show the warning,
    # then load without it.
    
    # Test with a dummy NER model path (will likely fail for most users unless they have it)
    dummy_ner_path = "C:/Users/beqa/MEHEDI/Notebook/ner all dataset/biobert_model_ner"
    print(f"Attempting to initialize model with NER weights from: {dummy_ner_path}")
    qa_model_with_ner = initialize_model(model_checkpoint_name, ner_model_path=dummy_ner_path)
    print(f"Model type (with dummy NER path attempt): {type(qa_model_with_ner)}")
    print("---")

    # Test initialization without NER model path
    print(f"Initializing model without NER weights.")
    qa_model_standalone = initialize_model(model_checkpoint_name)
    print(f"Model type (standalone): {type(qa_model_standalone)}")
    print(f"Model device: {qa_model_standalone.device}")

    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}")
    qa_model_standalone.to(device)
    print(f"Model device after moving: {qa_model_standalone.device}")

    # Example of saving a model
    # save_path = './checkpoints/my_qa_model'
    # qa_model_standalone.save_pretrained(save_path)
    # print(f"\nModel saved to: {save_path}")

