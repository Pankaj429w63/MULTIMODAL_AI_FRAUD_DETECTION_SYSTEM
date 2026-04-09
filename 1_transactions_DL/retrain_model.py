#!/usr/bin/env python
"""
Phase 2: Retrain Transaction Model with Extended Dataset
Uses the corrected feature pipeline and expanded data
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "transaction_model_v2.pt")
DATA_PATH = os.path.join(BASE_PATH, "data", "transactions_extended.csv")
RESULTS_DIR = os.path.join(BASE_PATH, "training_results")

os.makedirs(RESULTS_DIR, exist_ok=True)

class FraudModel(nn.Module):
    """Neural network for fraud detection"""
    def __init__(self, input_size=4, hidden_size=32):
        super(FraudModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_and_prepare_data():
    """Load transaction data and prepare for training"""
    print("[1] Loading transaction data...")
    
    df = pd.read_csv(DATA_PATH)
    print(f"    Loaded {len(df)} transactions")
    print(f"    Fraud ratio: {df['isFraud'].mean():.2%}")
    
    # Feature engineering (matching training pipeline exactly)
    print("[2] Feature engineering...")
    
    # Initialize encoders
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    # Encode transaction type
    df['type_encoded'] = label_encoder.fit_transform(df['type'])
    
    # Prepare numeric features
    numeric_features = df[['amount', 'oldbalanceOrg', 'newbalanceOrig']].values
    scaled_features = scaler.fit_transform(numeric_features)
    
    # Reconstruct 4-feature array in correct order: [amount, type, oldbal, newbal]
    X = np.zeros((len(df), 4))
    X[:, 0] = scaled_features[:, 0]  # amount (scaled)
    X[:, 1] = df['type_encoded'].values  # type (unscaled)
    X[:, 2] = scaled_features[:, 1]  # oldbalanceOrg (scaled)
    X[:, 3] = scaled_features[:, 2]  # newbalanceOrig (scaled)
    
    y = df['isFraud'].values
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    print(f"    Feature shape: {X_train.shape}")
    
    # Save encoders
    encoder_path = os.path.join(BASE_PATH, "label_encoder_v2.pkl")
    scaler_path = os.path.join(BASE_PATH, "scaler_v2.pkl")
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"    Saved encoders to v2 files")
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the fraud detection model"""
    print("[3] Training model...")
    
    device = torch.device('cpu')
    
    # Create data loaders
    train_tensor = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train.reshape(-1, 1))
    )
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    test_tensor = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test.reshape(-1, 1))
    )
    test_loader = DataLoader(test_tensor, batch_size=batch_size)
    
    # Initialize model
    model = FraudModel(input_size=4, hidden_size=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    return model, train_losses, test_losses, test_accuracies

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("[4] Evaluating model...")
    
    device = torch.device('cpu')
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_labels = (predictions > 0.5).float()
    
    accuracy = (predicted_labels.cpu().numpy().flatten() == y_test).mean()
    
    # Calculate precision, recall, F1
    tp = ((predicted_labels.cpu().numpy().flatten() == 1) & (y_test == 1)).sum()
    fp = ((predicted_labels.cpu().numpy().flatten() == 1) & (y_test == 0)).sum()
    fn = ((predicted_labels.cpu().numpy().flatten() == 0) & (y_test == 1)).sum()
    tn = ((predicted_labels.cpu().numpy().flatten() == 0) & (y_test == 0)).sum()
    
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

def save_model_and_visualizations(model, train_losses, test_losses, test_accuracies):
    """Save model and create training visualizations"""
    print("[5] Saving model and visualizations...")
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"    Saved model to: {MODEL_PATH}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(test_accuracies)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy Over Epochs')
    axes[1].grid()
    
    plot_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"    Saved plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 2: TRANSACTION MODEL RETRAINING")
    print("=" * 70 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, le, scaler = load_and_prepare_data()
    
    # Train model
    model, train_losses, test_losses, test_accs = train_model(X_train, y_train, X_test, y_test, epochs=50)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model_and_visualizations(model, train_losses, test_losses, test_accs)
    
    print("\n" + "=" * 70)
    print("TRANSACTION MODEL RETRAINING COMPLETE")
    print("=" * 70 + "\n")
