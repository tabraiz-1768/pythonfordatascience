"""
Startup Industry Classification Experiment
Implements the experiments described in the research abstract:
• TF-IDF + Linear SVM
• Word2Vec Embeddings + Logistic Regression
• BERT / RoBERTa Transformer Models (HuggingFace)

Dataset must contain:
 - One_Line_Pitch : startup description text
 - Industry       : one of 15 industry classes

Author: (You can add your name here)
"""

import re
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# Load dataset
DATA_PATH = "startup_company_one_line_pitches.csv"
TEXT_COL = "One_Line_Pitch"
LABEL_COL = "Industry"

df = pd.read_csv(DATA_PATH).dropna()

# Label encoding
lbl = LabelEncoder()
df["label"] = lbl.fit_transform(df[LABEL_COL])
num_classes = len(lbl.classes_)

# Split dataset
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)


def evaluate(name, y_true, y_pred):
    print(f"\n============ {name} ============")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1 Macro:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=lbl.classes_))


# ----------------------------------------------------
# MODEL 1 — TF-IDF + LinearSVC
# ----------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_df[TEXT_COL])
X_test = tfidf.transform(test_df[TEXT_COL])

svm = LinearSVC()
svm.fit(X_train, train_df["label"])
pred_svm = svm.predict(X_test)

evaluate("TF-IDF + Linear SVM", test_df["label"], pred_svm)


# ----------------------------------------------------
# MODEL 2 — Word2Vec + Logistic Regression
# ----------------------------------------------------
def tokenize(text): return re.findall(r"[a-zA-Z]+", text.lower())

sentences = train_df[TEXT_COL].apply(tokenize)
w2v = Word2Vec(sentences, vector_size=200, min_count=2, workers=4, sg=1, epochs=10)


def embed(doc):
    words = tokenize(doc)
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(200)


X_train_w2v = np.vstack(train_df[TEXT_COL].apply(embed))
X_test_w2v = np.vstack(test_df[TEXT_COL].apply(embed))

logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train_w2v, train_df["label"])
pred_w2v = logreg.predict(X_test_w2v)

evaluate("Word2Vec + Logistic Regression", test_df["label"], pred_w2v)


# ----------------------------------------------------
# MODEL 3 — Transformers (BERT / RoBERTa)
# ----------------------------------------------------
MODEL_NAME = "bert-base-uncased"  # switch to roberta-base to compare

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, padding="max_length", max_length=64)


train_encodings = train_df.apply(encode, axis=1, result_type="expand")
test_encodings = test_df.apply(encode, axis=1, result_type="expand")


class PitchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = torch.stack([torch.tensor(i["input_ids"]) for i in encodings])
        self.att_mask = torch.stack([torch.tensor(i["attention_mask"]) for i in encodings])
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.att_mask[idx],
            "labels": self.labels[idx]
        }

    def __len__(self): return len(self.labels)


train_ds = PitchDataset(train_encodings, train_df["label"].values)
test_ds = PitchDataset(test_encodings, test_df["label"].values)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes
)

training_args = TrainingArguments(
    output_dir="transformer_results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()
pred_logits = trainer.predict(test_ds).predictions
pred_transformer = np.argmax(pred_logits, axis=1)

evaluate("Transformer Model (BERT)", test_df["label"], pred_transformer)