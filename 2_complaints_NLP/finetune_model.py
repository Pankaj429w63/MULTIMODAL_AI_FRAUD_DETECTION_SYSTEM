#!/usr/bin/env python
"""
Phase 2: Fine-tune NLP Model on Fraud Complaints
Transfer learning from DistilBERT to fraud-specific classification
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "complaints.csv")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "complaint_model_v2")
RESULTS_DIR = os.path.join(BASE_PATH, "training_results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_complaint_data():
    """Load complaint dataset"""
    print("[1] Loading complaint data...")
    
    complaints = []
    labels = []
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            complaints.append(row['complaint_text'])
            labels.append(int(row['is_fraud']))
    
    print(f"    Loaded {len(complaints)} complaints")
    print(f"    Fraud ratio: {sum(labels)/len(labels):.2%}")
    
    # Split into train/val/test
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        complaints, labels, test_size=0.2, random_state=42, stratify=labels
    )
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train, labels_train, test_size=0.2, random_state=42, stratify=labels_train
    )
    
    print(f"    Train: {len(texts_train)}, Val: {len(texts_val)}, Test: {len(texts_test)}")
    
    return {
        'train': (texts_train, labels_train),
        'val': (texts_val, labels_val),
        'test': (texts_test, labels_test)
    }

def create_datasets(texts_train, labels_train, texts_val, labels_val):
    """Create HuggingFace datasets"""
    print("[2] Creating HuggingFace datasets...")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(texts, labels):
        tokens = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'labels': torch.tensor(labels)
        }
    
    train_dataset = Dataset.from_dict({
        'text': texts_train,
        'label': labels_train
    })
    
    val_dataset = Dataset.from_dict({
        'text': texts_val,
        'label': labels_val
    })
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
        remove_columns=['text']
    )
    train_dataset = train_dataset.rename_column('label', 'labels')
    train_dataset.set_format('torch')
    
    val_dataset = val_dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
        remove_columns=['text']
    )
    val_dataset = val_dataset.rename_column('label', 'labels')
    val_dataset.set_format('torch')
    
    print(f"    Train dataset: {len(train_dataset)}")
    print(f"    Val dataset: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer

def train_nlp_model(train_dataset, val_dataset):
    """Fine-tune DistilBERT on fraud data"""
    print("[3] Fine-tuning NLP model...")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(MODEL_SAVE_PATH)
    print(f"    Saved model to: {MODEL_SAVE_PATH}")
    
    return model, trainer

def evaluate_nlp(model, tokenizer, test_texts, test_labels):
    """Evaluate NLP model on test set"""
    print("[4] Evaluating NLP model...")
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    
    with torch.no_grad():
        for text, label in zip(test_texts, test_labels):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=128
            ).to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            
            all_preds.append(pred)
            if pred == label:
                correct += 1
            total += 1
    
    accuracy = correct / total
    
    # Calculate metrics
    tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, test_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, test_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, test_labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, test_labels))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"      TN: {tn}, FP: {fp}")
    print(f"      FN: {fn}, TP: {tp}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 2: NLP MODEL FINE-TUNING")
    print("=" * 70 + "\n")
    
    # Load data
    data_splits = load_complaint_data()
    texts_train, labels_train = data_splits['train']
    texts_val, labels_val = data_splits['val']
    texts_test, labels_test = data_splits['test']
    
    # Create datasets
    train_ds, val_ds, tokenizer = create_datasets(texts_train, labels_train, texts_val, labels_val)
    
    # Train
    model, trainer = train_nlp_model(train_ds, val_ds)
    
    # Evaluate
    metrics = evaluate_nlp(model, tokenizer, texts_test, labels_test)
    
    print("\n" + "=" * 70)
    print("NLP MODEL FINE-TUNING COMPLETE")
    print("=" * 70 + "\n")
