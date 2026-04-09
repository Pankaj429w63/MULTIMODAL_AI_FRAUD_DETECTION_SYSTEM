import pandas as pd

def preprocess_data(input_path="Complaints.csv"):
    df = pd.read_csv(input_path, low_memory=False)

    df = df[["Consumer complaint narrative", "Product"]].dropna()
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].str.lower()

    df["label"] = df["Product"].apply(
        lambda x: 1 if "fraud" in str(x).lower() or "unauthorized" in str(x).lower() else 0
    )

    df.rename(columns={"Consumer complaint narrative": "complaint_text"}, inplace=True)

    df.to_csv("processed_complaints.csv", index=False)
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    preprocess_data()
