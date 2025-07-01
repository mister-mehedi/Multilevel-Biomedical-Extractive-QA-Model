# 🧠 Multilevel Biomedical Extractive QA Model

A powerful deep learning framework designed to enhance extractive Question Answering (QA) in the **biomedical domain**. Built on BioBERT and enriched with custom NER encoder integration, multilevel preprocessing, and a rich suite of evaluation metrics—including Exact Match, F1, MRR, Lacc, and Sacc.

---

## ✨ Features

- 💊 **Domain-specific optimization** using [BioBERT v1.1](https://huggingface.co/dmis-lab/biobert-v1.1)
- 🧠 **Multilevel metrics**: EM, F1, MRR, Lacc (Local Accuracy), Sacc (Sequence Accuracy)
- 🔁 **Bootstrap training** for model ensembles or bagging strategies
- ⚙️ **NER and RE model integration** to inject pretrained biomedical knowledge
- 📊 Clean performance visualizations with detailed metric breakdowns


---

## 🛠 Installation

```bash
# Optionally create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

requirements.txt:
```
torch
transformers
datasets
scikit-learn
numpy
matplotlib
seaborn
tqdm
accelerate
evaluate
```


---

🚀 How to Run
```
python main.py
```

This triggers a full pipeline:
- Loads BioASQ-formatted .json data
- Preprocesses it for training and validation
- Initializes BioBERT (optionally loads NER & RE encoder)
- Trains the QA model
- Evaluates using rich QA metrics
- Saves the model and results in checkpoints/
Default configuration is customizable in main.py

---


🧹 Preprocessing Pipeline
The scripts/preprocessor.py module ensures high-quality input preparation:
- Tokenizes questions and contexts with sliding window truncation
- Computes accurate start and end positions for training
- Applies robust offset mapping for metric computation
- Includes a custom CustomDataset wrapper for PyTorch integration


---


🧠 Model Initializatio
```
from scripts.model import initialize_model

model = initialize_model(
    model_checkpoint="dmis-lab/biobert-v1.1",
    ner_model_path="./path_to_ner_model"
    re_model_path="./path_to_re_model"
)
```
Loads a QA-capable BioBERT model and optionally transfers weights from a NER-trained encoder. Useful for biomedical transfer learning.

---

📦 Data Format
Input data must follow this simplified SQuAD/BioASQ format
```
{
  "data": [
    {
      "paragraphs": [
        {
          "context": "Relevant biomedical passage here...",
          "qas": [
            {
              "id": "bio-123",
              "question": "What is the role of X?",
              "answers": [{"text": "X is a protein", "answer_start": 18}]
            }
          ]
        }
      ]
    }
  ]
}
```

---


## 📊 Evaluation Metrics

| Metric           | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| **F1 Score**     | Measures the overlap between predicted and true answers                |
| **Exact Match**  | Assesses whether the predicted answer matches exactly with the ground truth |
| **MRR**          | Mean Reciprocal Rank – scores based on position of correct answer      |
| **Lacc**         | Local Accuracy – token-level accuracy at predicted start and end       |
| **Sacc**         | Sequence Accuracy – assesses full predicted text against true answer   |

---

