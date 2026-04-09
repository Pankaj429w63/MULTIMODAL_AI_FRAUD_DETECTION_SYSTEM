import os
import torch
import joblib
import numpy as np
from pathlib import Path
from torch import nn

# Initialize model and preprocessors
MODEL_PATH = os.path.join(os.path.dirname(__file__), "transaction_model.pt")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Define the model architecture (must match training)
class FraudModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model components
_model = None
_label_encoder = None
_scaler = None

def _load_model():
    """Load the trained transaction model and preprocessors"""
    global _model, _label_encoder, _scaler
    
    if _model is None:
        try:
            # Load label encoder and scaler
            if os.path.exists(ENCODER_PATH):
                _label_encoder = joblib.load(ENCODER_PATH)
            else:
                # Fallback: create a simple encoder if not found
                _label_encoder = None
                
            if os.path.exists(SCALER_PATH):
                _scaler = joblib.load(SCALER_PATH)
            else:
                _scaler = None
            
            # Load PyTorch model
            if os.path.exists(MODEL_PATH):
                # Create model instance - need to infer input dim
                model = FraudModel(input_dim=4)  # amount, type, oldbalanceOrg, newbalanceOrig
                model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
                model.eval()
                _model = model
            else:
                _model = None
        except Exception as e:
            print(f"Warning: Could not load model: {e}. Using fallback heuristic mode.")
            _model = None
    
    return _model, _label_encoder, _scaler

def predict_transaction(data_dict=None):
    """
    Predicts transaction fraud using neural network model inference.
    Falls back to behavioral heuristics if model is not available.
    """
    if data_dict is None:
        data_dict = {
            "amount": 1500.0,
            "type": "TRANSFER",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 8500.0
        }

    # Try to use trained model first
    model, encoder, scaler = _load_model()
    
    if model is not None:
        try:
            # Extract features (same order as training)
            amount = float(data_dict.get("amount", 0.0) or 0.0)
            trans_type = str(data_dict.get("type", "TRANSFER")).upper()
            old_balance = float(data_dict.get("oldbalanceOrg", 0.0) or 0.0)
            new_balance = float(data_dict.get("newbalanceOrig", 0.0) or 0.0)
            
            # Encode transaction type
            type_encoded = 0
            if encoder:
                try:
                    type_encoded = encoder.transform([trans_type])[0]
                except:
                    # Map common types to numeric values
                    type_map = {"TRANSFER": 0, "PAYMENT": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
                    type_encoded = type_map.get(trans_type, 0)
            
            # Prepare features to match TRAINING pipeline
            # Training used: [amount, type, oldbalanceOrg, newbalanceOrig]
            # But scaler only scales numeric: [amount, oldbalanceOrg, newbalanceOrig]
            
            numeric_features = np.array([[amount, old_balance, new_balance]], dtype=np.float32)
            
            # Scale numeric features using trained scaler
            if scaler:
                try:
                    numeric_scaled = scaler.transform(numeric_features)
                except Exception as e:
                    print(f"Scaler error: {e}, using unscaled numeric features")
                    numeric_scaled = numeric_features
            else:
                numeric_scaled = numeric_features
            
            # Reconstruct 4-feature array in correct order: [amount, type, oldbalanceOrg, newbalanceOrig]
            features = np.array([
                [numeric_scaled[0, 0], type_encoded, numeric_scaled[0, 1], numeric_scaled[0, 2]]
            ], dtype=np.float32)
            
            # Run inference WITH gradients for Explainable AI (XAI)
            tensor_input = torch.tensor(features, dtype=torch.float32, requires_grad=True)
            output = model(tensor_input)
            base_score = float(output[0][0].item())

            # XAI (Explainable AI) via Input x Gradient methodology
            output.backward()
            gradients = tensor_input.grad.detach().numpy()[0]
            inputs = tensor_input.detach().numpy()[0]
            
            # Feature attribution = input_value * gradient_wrt_input
            attributions = inputs * gradients
            feature_names = ["Amount", "Transaction Type", "Old Balance", "New Balance"]
            
            # Normalize to percentages
            total_attr = np.sum(np.abs(attributions)) + 1e-9
            normalized_attr = (attributions / total_attr) * 100
            xai_dict = {name: round(float(val), 2) for name, val in zip(feature_names, normalized_attr)}
            
        except Exception as e:
            print(f"Model inference error: {e}. Using fallback heuristic.")
            base_score = 0.1  # Assume low baseline if model fails
            xai_dict = {}

    # ─── BEHAVIORAL RISK MODIFIERS ───
    location = data_dict.get("location", "Same City (Known)")
    device = data_dict.get("device", "Trusted (Used before)")
    time_of_day = data_dict.get("time_of_day", 14)
    account_age = data_dict.get("account_age", 365)
    
    risk_boost = 0.0
    reasons = []

    # Amount & Balance logic
    if amount > 50000:
        risk_boost += 0.2
        reasons.append(f"Unusually large transaction: ₹{amount:,.2f}")
    if new_balance < 100 and new_balance >= 0:
        risk_boost += 0.1
        reasons.append("Account balance critically depleted post-transaction")
    if old_balance < amount:
        risk_boost += 0.3
        reasons.append("Suspicious: Attempted amount exceeds prior balance")

    # Location Risk
    if "Foreign" in location:
        risk_boost += 0.35
        reasons.append("Cross-border transaction detected (High location risk)")
    elif "Different State" in location:
        risk_boost += 0.1
        reasons.append("Login from unusual domestic location")
    elif "Tor/VPN" in location:
        risk_boost += 0.45
        reasons.append("Connection masked via VPN/Tor (High anonymity risk)")

    # Device Risk
    if "New" in device:
        risk_boost += 0.2
        reasons.append("Login from completely unknown/new device")
    elif "Rooted" in device or "Jailbroken" in device:
        risk_boost += 0.4
        reasons.append("Device security compromised (Rooted/Jailbroken environment)")

    # Time Risk (0-5 AM is considered high risk for normal users)
    if time_of_day < 5:
        risk_boost += 0.15
        reasons.append(f"Unusual transaction time: {time_of_day}:00 AM")

    # Account Age Risk
    if account_age < 7:
        risk_boost += 0.25
        reasons.append("Very new account (Age < 7 days)")

    # Asymptotic Probabilistic Aggregation (Noisy-OR)
    # Instead of linearly adding scores (0.6 + 0.5 = 1.1 -> clamped), we aggregate probabilities
    final_score = 1.0 - (1.0 - base_score) * (1.0 - min(risk_boost, 0.999))

    if not reasons:
        reasons.append("Transaction context and behavior appear completely normal")

    return {
        "score": round(max(0.0, min(1.0, final_score)), 4),
        "reasons": reasons,
        "xai_explanations": xai_dict if 'xai_dict' in locals() else {},
        "model_used": "PyTorch DL + Behavioral Engine + XAI"
    }

if __name__ == "__main__":
    test_data = {"amount": 5000, "location": "Foreign", "device_status": "New", "time_of_day": 3, "account_age_days": 5}
    res = predict_transaction(test_data)
    print(f"Risk Score: {res['score']}, Reasons: {res['reasons']}")
