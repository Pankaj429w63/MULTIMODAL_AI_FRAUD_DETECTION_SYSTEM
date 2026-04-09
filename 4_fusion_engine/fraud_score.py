def final_fraud_score(transaction_res, complaint_res, identity_res):
    """
    Aggregates scores and reasons from all vectors to produce a final report.
    """
    t_score = transaction_res['score']
    c_score = complaint_res['score']
    i_score = identity_res['score'] # trust score
    
    # Invert identity trust to risk
    i_risk = 1 - i_score
    
    # Adaptive Dynamic Weighting (Context-Aware Fusion)
    # Start with baseline normal weights
    w_t, w_c, w_i = 0.50, 0.25, 0.25
    
    # 1. Identity Crisis Modifier
    # If the user's selfie completely fails to match the ID card, it doesn't matter
    # if the transaction amount "looks normal." This is the highest risk scenario.
    if i_risk > 0.80:
        w_i = 0.70  # Shift 70% of the entire decision weight to the Computer Vision failure
        w_t = 0.20
        w_c = 0.10
        
    # 2. Panic Narrative Modifier
    # If the NLP model detects high panic (e.g., "my account was hacked"), 
    # we heavily weigh the text, even if the transaction amount is tiny.
    elif c_score > 0.80:
        w_c = 0.60  # Shift 60% of weight to NLP Transformer
        w_t = 0.20
        w_i = 0.20
        
    # Final Continuous Fusion Calculation
    final_score = (w_t * t_score) + (w_c * c_score) + (w_i * i_risk)
    
    # Collect all risk reasons
    all_reasons = []
    
    # Add transaction reasons if they contributed to risk
    if t_score > 0.2:
        all_reasons.extend([f"[Transaction] {r}" for r in transaction_res['reasons']])
        
    # Add NLP reasons
    if c_score > 0.2:
        all_reasons.extend([f"[Sentiment] {r}" for r in complaint_res['reasons']])
        
    # Add identity reasons (especially if low trust)
    if i_score < 0.7:
        all_reasons.extend([f"[Identity] {r}" for r in identity_res['reasons']])

    # AI Red Flag Overrides
    if t_score > 0.85:
        final_score = max(final_score, 0.90)
        all_reasons.append("🚨 CRITICAL: Extreme Transaction Deep Learning Risk")
    if c_score > 0.85:
        final_score = max(final_score, 0.90)
        all_reasons.append("🚨 CRITICAL: High-Panic NLP Transformer Narrative")
    if i_risk > 0.85:
        final_score = max(final_score, 0.98)
        all_reasons.append("🚨 CRITICAL DEEPFAKE OR STOLEN ID: Swin Vector completely failed")

    # Decision thresholds
    # Tightened 'Review' window: <0.65 is strictly LEGIT, >0.85 is rigorously FRAUD
    if final_score > 0.85:
        decision = "FRAUD"
    elif final_score > 0.75:
        decision = "POSSIBLY FRAUD"
    elif final_score > 0.65:
        decision = "POSSIBLY LEGIT"
    else:
        decision = "LEGIT"

    # Ensure reasons is never empty
    if not all_reasons:
        all_reasons.append("No significant risk factors identified across analysis vectors")

    return {
        "final_score": round(float(final_score), 4),
        "decision": decision,
        "risk_factors": all_reasons
    }

if __name__ == "__main__":
    t = {"score": 0.8, "reasons": ["Large amount", "Balance mismatch"]}
    c = {"score": 0.1, "reasons": ["No keywords"]}
    i = {"score": 0.9, "reasons": ["Verified"]}
    
    print(final_fraud_score(t, c, i))
