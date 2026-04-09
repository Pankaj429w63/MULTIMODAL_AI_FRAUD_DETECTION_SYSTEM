import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("processed_complaints.csv")
sample_size = min(1000, len(df))
df = df.sample(n=sample_size, random_state=42)

MODEL_NAME = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2, 
    ignore_mismatched_sizes=True
)

encodings = tokenizer(
    df["complaint_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

labels = torch.tensor(df["label"].values)
dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# STABILITY FIX 1: Use a slightly lower learning rate for DeBERTa
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# STABILITY FIX 2: Add a Warmup Scheduler
# This prevents the model from taking huge steps before the new classifier head is ready
num_epochs = 3
total_steps = len(loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1 * total_steps), # 10% warmup
    num_training_steps=total_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
print("Starting training...")

for epoch in range(num_epochs):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # STABILITY FIX 3: Check for NaN before stepping
        if torch.isnan(loss):
            print("NaN loss detected! Skipping batch...")
            continue
            
        loss.backward()
        
        # STABILITY FIX 4: Gradient Clipping
        # This prevents "exploding gradients" which is the main cause of NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step() # Don't forget to step the scheduler!

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

model.save_pretrained("complaint_model")
tokenizer.save_pretrained("complaint_model")
print("Complaint NLP model trained & saved successfully")
