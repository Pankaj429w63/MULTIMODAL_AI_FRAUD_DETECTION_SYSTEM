import os
import math

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Note: Nested structure observed in complaint_model directory
MODEL_PATH = os.path.join(BASE_PATH, "complaint_model/complaint_model")

# Set to True only if you want to use the heavy transformer model
USE_HEAVY_MODEL = True

def predict_complaint(text):
    """
    Predicts fraud risk from complaint text and returns score plus reasoning.
    Uses sentiment classification as a proxy for fraud detection.
    """
    reasons = []
    
    if USE_HEAVY_MODEL:
        try:
            from transformers import pipeline
            
            # Use simple pipeline for sentiment analysis with strict token truncation
            # This prevents crashes when users paste massive text blocks > 512 tokens
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512
            )
            
            # Predict securely with full text (pipeline handles the token bounding)
            result = sentiment_pipeline(text)[0]
            
            # Map sentiment to fraud risk
            # NEGATIVE = more likely fraud (0.7-0.9)
            # POSITIVE = less likely fraud (0.0-0.3)  
            if result['label'] == 'NEGATIVE':
                base_score = 0.8
            else:
                base_score = 0.2
                
            # Adjust by confidence
            confidence = result['score']  # 0.5-1.0
            raw_score = base_score * confidence
            
            # ADJUSTMENT: Apply conservative scaling
            score = raw_score * 0.7  # Reduce urgency
            
            reasons.append(f"Complaint sentiment: {result['label']} (confidence: {confidence:.2%}), fraud risk: {score:.2%}")
            return {"score": max(0.0, min(1.0, score)), "reasons": reasons}
        except Exception as e:
            print(f"NLP model error: {e}, falling back to keywords")
            pass

    # HEURISTIC FALLBACK
    t = (text or "").lower()
    keywords = {
        "fraud": 0.3,
        "unauthorized": 0.3,
        "scam": 0.2,
        "stolen": 0.2,
        "hacked": 0.3,
        "phishing": 0.3,
        "suspicious": 0.1,
        "urgent": 0.1,
        "immediately": 0.1,
        "compromised": 0.2
    }
    
    risk = 0.1 # base risk
    hits = []
    for word, weight in keywords.items():
        if word in t:
            risk += weight
            hits.append(word)
            
    if hits:
        reasons.append(f"Detected suspicious keywords: {', '.join(hits)}")
    else:
        reasons.append("No suspicious keywords detected in narrative")
        
    return {
        "score": round(max(0.0, min(1.0, risk)), 4),
        "reasons": reasons
    }

if __name__ == "__main__":
    test_text = "Money was deducted without my permission. My account was hacked."
    res = predict_complaint(test_text)
    print(f"Score: {res['score']}, Reasons: {res['reasons']}")
