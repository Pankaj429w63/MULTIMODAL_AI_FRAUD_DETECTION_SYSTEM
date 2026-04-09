import os
import sys
import tempfile
import json
from fastapi import FastAPI, HTTPException, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse

# ─────────────────────────────────────────────
# 1. Setup ML Models path
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBDIRS = ["1_transactions_DL", "2_complaints_NLP", "3_kyc_CV", "4_fusion_engine"]
for sd in SUBDIRS:
    path = os.path.join(BASE_DIR, sd)
    if path not in sys.path:
        sys.path.insert(0, path)

import importlib.util
def load_predictor(subdir, name, filename="predict.py"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE_DIR, subdir, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────
# 2. Globally Load Models Into Memory Once
# ─────────────────────────────────────────────
print("Loading heavy PyTorch & NLP models into API memory... This takes ~10 seconds.")
tx_mod = load_predictor("1_transactions_DL", "tx_predict")
nlp_mod = load_predictor("2_complaints_NLP", "nlp_predict")
cv_mod = load_predictor("3_kyc_CV", "cv_predict")
fe_mod = load_predictor("4_fusion_engine", "fraud_score", "fraud_score.py")
print("[OK] Models loaded successfully. API is ready.")

# ─────────────────────────────────────────────
# 3. FastAPI Service Definition
# ─────────────────────────────────────────────
app = FastAPI(title="AI Fraud Detection Engine - Independent Microservice")

@app.get("/")
async def home():
    return {"message": "Fraud Detection Microservice API (FastAPI) is live."}

@app.post("/predict")
async def analyze_fraud(
    amount: float = Form(...),
    type: str = Form(...),
    oldbalanceOrg: float = Form(...),
    newbalanceOrig: float = Form(...),
    location: str = Form(...),
    device: str = Form(...),
    time_of_day: int = Form(...),
    account_age: int = Form(...),
    complaint_text: str = Form(...),
    selfie_file: UploadFile = File(None),
    id_file: UploadFile = File(None)
):
    """
    Microservice endpoint exposing Deep Learning + NLP + Computer Vision.
    No UI blocking here!
    """
    try:
        # 1. Transaction Behavioral Analysis (PyTorch + Heuristics + XAI)
        tx_data = {
            "amount": amount, "type": type,
            "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
            "location": location, "device": device,
            "time_of_day": time_of_day, "account_age": account_age
        }
        tx_res = tx_mod.predict_transaction(tx_data)
        
        # 2. NLP Sentiment Analysis via Transformers
        nlp_res = nlp_mod.predict_complaint(complaint_text)

        # 3. Computer Vision / KYC
        # Defaults if no images uploaded
        s_path = "3_kyc_CV/data/Selfies ID Images dataset/18_sets_ Caucasians/0001ca9b9a--61adf4903e0f222c5a048507_age_20_name_Kasia/Selfie_1.jpg"
        i_path = "3_kyc_CV/data/Selfies ID Images dataset/18_sets_ Caucasians/0001ca9b9a--61adf4903e0f222c5a048507_age_20_name_Kasia/ID_1.jpg"
        
        temp_files = []
        if selfie_file and id_file:
            fd_s, s_path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd_s, 'wb') as f:
                f.write(await selfie_file.read())
                
            fd_i, i_path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd_i, 'wb') as f:
                f.write(await id_file.read())
            temp_files.extend([s_path, i_path])

        cv_res = cv_mod.predict_kyc(s_path, i_path)

        # Cleanup OS temp files immediately to prevent memory/hdd leaks
        for tmp in temp_files:
            try: os.remove(tmp)
            except: pass

        # 4. Fusion Engine aggregation
        fusion_result = fe_mod.final_fraud_score(tx_res, nlp_res, cv_res)

        return {
            "status": "success",
            "t_result": tx_res,
            "c_result": nlp_res,
            "i_result": cv_res,
            "fusion_result": fusion_result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"API Processing Error: {str(e)}")
