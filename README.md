# named_entity_recognition.ipynb

This project fine‑tunes **DistilBERT (uncased)** for **Named Entity Recognition (NER)** using Hugging Face Transformers and PyTorch Lightning. It trains on token‑labeled sentences and predicts entity tags per token (e.g., people, organizations, locations, times).

---

## 🔹 Motivation
NER converts unstructured text into structured information. It’s widely used for:

- Document understanding (contracts, research papers)

- Search & recommendation (entity‑aware retrieval)

- Customer support / CRM (extracting dates, names, orgs)

---

## 🔹 Dataset
- Source: **DeepLearning.AI**.

**data**:
```
- train_sentences.txt
- train_labels.txt
- val_sentences.txt
- val_labels.txt
- test_sentences.txt
- test_labels.txt
``` 

**Tag set** used in the notebook:
```
['B-art','B-eve','B-geo','B-gpe','B-nat','B-org','B-per','B-tim',
'I-art','I-eve','I-geo','I-gpe','I-nat','I-org','I-per','I-tim','O']
```

---

## 🔹 Methodology
- Tokenizer: AutoTokenizer.from_pretrained("distilbert-base-uncased")

- Model: AutoModelForTokenClassification with a classification head (num_labels = 17)

- Collation: DataCollatorForTokenClassification for dynamic padding

- Lightning: LitTokenClassifier implements training_step, validation_step, and configure_optimizers

- Prediction helper: predict(model, tokenizer, words) aligns sub‑tokens back to word‑level predictions

---

## 🔹 How to Run
```bash
# Clone the repository
git clone <repo_url>
cd named_entity_recognition.ipynb

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter named_entity_recognition.ipynb
```

---

## 🔹 Requirements
The main dependencies are listed in `requirements.txt`:

```
torch
transformers
pytorch_lightning
pandas
jupyter
```

- Install them with:
pip install -r requirements.txt


---


## 🔹 Skills Demonstrated
- Token‑level sequence labeling (NER)

- Hugging Face Transformers for token classification

- Clean training loops with PyTorch Lightning

- Data collation & label alignment for subword tokenization

- Qualitative debugging of predictions


---

## 🔹 License
This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

