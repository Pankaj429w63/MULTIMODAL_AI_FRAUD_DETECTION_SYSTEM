import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv("data/online_payments.csv")

# Keep useful columns
df = df[['amount', 'type', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']]

# Encode categorical
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Scale numeric
scaler = StandardScaler()
num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
df[num_cols] = scaler.fit_transform(df[num_cols])

joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

df.to_csv("data/processed.csv", index=False)
print("Transaction preprocessing complete")


