import streamlit as st
import os
import sys
import importlib.util
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time
import json
from datetime import datetime
from io import BytesIO
import tempfile

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DB import (after page config)
# ─────────────────────────────────────────────
import database as db

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0f1e 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2d40;
}

.hero-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 40px rgba(0,120,255,0.08);
}
.hero-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #64748b; font-size: 0.9rem; margin: 4px 0 0; }

.card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.card-title { font-size: 1rem; font-weight: 600; color: #60a5fa; margin-bottom: 16px; }

.decision-fraud {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444; border-radius: 14px; padding: 24px; text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.2); animation: pulse-red 2s infinite;
}
.decision-review {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1px solid #f59e0b; border-radius: 14px; padding: 24px; text-align: center;
    box-shadow: 0 0 40px rgba(245,158,11,0.2);
}
.decision-legit {
    background: linear-gradient(135deg, #14532d, #166534);
    border: 1px solid #22c55e; border-radius: 14px; padding: 24px; text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.15);
}
.decision-text  { font-size: 2rem; font-weight: 800; color: white; letter-spacing: 0.1em; }
.decision-subtitle { font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-top: 6px; }

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 30px rgba(239,68,68,0.2); }
    50%       { box-shadow: 0 0 60px rgba(239,68,68,0.5); }
}

.risk-item {
    background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444;
    border-radius: 6px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.875rem; color: #fca5a5;
}
.info-item {
    background: rgba(34,197,94,0.08); border-left: 3px solid #22c55e;
    border-radius: 6px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.875rem; color: #86efac;
}

.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 20px 0;
}

.sidebar-stat {
    background: rgba(30,58,95,0.3); border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 12px; margin: 8px 0; text-align: center;
}

.badge-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.15); border: 1px solid #22c55e;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #22c55e; font-weight: 600;
}
.dot-live {
    width: 7px; height: 7px; background: #22c55e; border-radius: 50%;
    animation: blink 1.2s infinite; display: inline-block;
}
@keyframes blink { 0%, 100% { opacity:1; } 50% { opacity:0.3; } }

/* Log table tweaks */
.log-row-fraud  { color: #ef4444 !important; font-weight: 700; }
.log-row-review { color: #f59e0b !important; font-weight: 700; }
.log-row-legit  { color: #22c55e !important; font-weight: 700; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API Endpoint Integration
# ─────────────────────────────────────────────
import requests
API_URL = "http://localhost:8000/predict"

try:
    # Just a simple health check to see if the microservice is running
    res = requests.get(API_URL.replace("/predict", "/"), timeout=5)
    models_ok = res.status_code == 200
except:
    st.error("❌ Cannot connect to FastAPI Microservice! Please start it in a separate terminal: `uvicorn api:app --port 8000`")
    models_ok = False

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div>
        <p class="hero-title">🛡️ AI Fraud Detection System</p>
        <p class="hero-sub">Multi-modal fraud analysis · Deep Learning · NLP · Computer Vision · Fusion Engine</p>
    </div>
    <div style="margin-left:auto">
        <span class="badge-live"><span class="dot-live"></span>LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
stats = db.get_stats()

with st.sidebar:
    st.markdown("## ⚙️ System Status")
    st.markdown(f"""
    <div class="sidebar-stat">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase">Models</div>
        <div style="color:{'#22c55e' if models_ok else '#ef4444'};font-weight:600">{'✓ 4 / 4 Loaded' if models_ok else '✗ Error'}</div>
    </div>
    <div class="sidebar-stat">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase">Total Analyses</div>
        <div style="color:#a78bfa;font-weight:700;font-size:1.2rem">{stats.get('total',0)}</div>
    </div>
    <div class="sidebar-stat">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase">Fraud / Review / Legit</div>
        <div style="font-weight:600">
            <span style="color:#ef4444">{stats.get('fraud_count',0)}</span> /
            <span style="color:#f59e0b">{stats.get('review_count',0)}</span> /
            <span style="color:#22c55e">{stats.get('legit_count',0)}</span>
        </div>
    </div>
    <div class="sidebar-stat">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase">Avg Fusion Score</div>
        <div style="color:#60a5fa;font-weight:700">{stats.get('avg_score') or '—'}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎛️ Thresholds")
    fraud_threshold  = st.slider("Fraud Threshold",  0.75, 1.00, 0.85, 0.01)
    suspicious_threshold = st.slider("Suspicious Threshold", 0.50, 0.85, 0.65, 0.01)

    st.markdown("---")
    st.markdown("## 📋 Recent Runs")
    recent = db.fetch_logs(limit=5)
    if recent:
        for h in recent:
            color = "#ef4444" if h["decision"]=="FRAUD" else ("#f59e0b" if h["decision"]=="REVIEW" else "#22c55e")
            st.markdown(f"""
            <div style="border-left:3px solid {color};padding:6px 10px;margin:4px 0;
                        background:rgba(255,255,255,0.03);border-radius:4px;font-size:0.78rem;">
                <span style="color:{color};font-weight:700">#{h['id']} {h['decision']}</span>
                <span style="color:#64748b"> — {h['final_score']:.2f}</span><br>
                <span style="color:#475569;font-size:0.7rem">{h['timestamp']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No analyses yet.")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_analyze, tab_logs, tab_analytics = st.tabs(["🔍 Analyze", "🗄️ Database Logs", "📈 Analytics"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Analyze
# ═══════════════════════════════════════════════════════════
with tab_analyze:

    # ── Plotly helpers ──────────────────────────────────────
    def make_gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%", "font": {"size": 26, "color": "#e2e8f0"}},
            title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569", "size": 10}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#1e293b",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(34,197,94,0.12)"},
                    {"range": [40, 75], "color": "rgba(245,158,11,0.12)"},
                    {"range": [75,100], "color": "rgba(239,68,68,0.12)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": round(value*100,1)},
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=20,r=20,t=50,b=10), height=220, font={"family":"Inter"})
        return fig

    def make_radar(t, c, i):
        categories = ["Transaction<br>Risk","Complaint<br>Risk","Identity<br>Risk","Fusion<br>Score","Overall<br>Threat"]
        fusion  = 0.5*t + 0.3*c + 0.2*(1-i)
        overall = min((t + c + (1-i)) / 3, 1.0)
        values  = [t, c, 1-i, fusion, overall]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values+[values[0]], theta=categories+[categories[0]],
            fill="toself", fillcolor="rgba(96,165,250,0.15)",
            line={"color": "#60a5fa", "width": 2},
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1],
                                tickfont={"color":"#475569","size":9},
                                gridcolor="#1e2d40", linecolor="#1e2d40"),
                angularaxis=dict(tickfont={"color":"#94a3b8","size":10},
                                 gridcolor="#1e2d40", linecolor="#1e2d40"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=40,t=30,b=30), height=280,
            showlegend=False, font={"family":"Inter"},
        )
        return fig

    def make_score_bar(scores_dict):
        labels = list(scores_dict.keys())
        vals   = [v*100 for v in scores_dict.values()]
        colors = ["#ef4444" if v>75 else "#f59e0b" if v>40 else "#22c55e" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont={"color":"#94a3b8","size":12},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=60,t=10,b=10), height=160,
            xaxis=dict(range=[0,115], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont={"color":"#94a3b8","size":12}, gridcolor="#1e2d40"),
            font={"family":"Inter"},
        )
        return fig

    # ── User Context ──────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">👤 Stateful User Identity</div>', unsafe_allow_html=True)
    username = st.text_input("Username (Case Insensitive)", value="", placeholder="Enter username to load saved KYC...")
    saved_kyc = None
    if username.strip() != "":
        saved_kyc = db.fetch_last_kyc(username.strip())
        if saved_kyc:
            st.success(f"✓ Verified returning user. KYC documents for **{username}** loaded from database. (You can skip uploading images below)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input Form ──────────────────────────────────────────
    col_left, col_right = st.columns([1,1], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">💳 Transaction Context & Behavior</div>', unsafe_allow_html=True)
        c0, c1 = st.columns(2)
        amount      = c0.number_input("Transaction Amount (₹)", min_value=0.0, value=15000.0, step=500.0)
        trans_type  = c1.selectbox("Transaction Type", ["TRANSFER","PAYMENT","CASH_OUT","DEBIT","CASH_IN"])
        
        c2, c3 = st.columns(2)
        old_balance = c2.number_input("Old Balance (₹)", min_value=0.0, value=20000.0, step=1000.0)
        new_balance = c3.number_input("New Balance (₹)", min_value=0.0, value=5000.0,  step=1000.0)

        c4, c5 = st.columns(2)
        location = c4.selectbox("Login Location", ["Same City (Known)", "Different State", "Foreign Country (High Risk)", "Tor/VPN Node"])
        device   = c5.selectbox("Device Status", ["Trusted (Used before)", "New Device", "Rooted / Jailbroken"])

        c6, c7 = st.columns(2)
        time_of_day = c6.slider("Time of Transaction (24h format)", min_value=0, max_value=23, value=14)
        acc_age     = c7.number_input("Account Age (Days)", min_value=0, value=365)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">💬 Customer Complaint (NLP)</div>', unsafe_allow_html=True)
        complaint_text = st.text_area(
            "Complaint", label_visibility="collapsed",
            value="",
            placeholder="Type customer complaint here (e.g., I suspect unauthorized access to my account...)",
            height=110,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card"><div class="card-title">🪪 Identity Verification (KYC)</div>', unsafe_allow_html=True)
        
        selfie_mode = st.radio("Selfie Input Method", ["Upload File", "Take Live Photo"], horizontal=True, label_visibility="collapsed")
        if selfie_mode == "Upload File":
            selfie_file = st.file_uploader("Upload Selfie",  type=["jpg","png","jpeg"], key="selfie_file")
        else:
            selfie_file = st.camera_input("Take Live Selfie", key="selfie_cam")
            
        id_file = st.file_uploader("Upload ID Card", type=["jpg","png","jpeg"], key="idcard")

        if selfie_file and id_file:
            p1, p2 = st.columns(2)
            p1.image(selfie_file, caption="Selfie")
            p2.image(id_file,     caption="ID Card")
            st.success("✓ New Images loaded — will be saved to database")
        elif saved_kyc:
            p1, p2 = st.columns(2)
            p1.image(saved_kyc["selfie"][1], caption="DB Selfie")
            p2.image(saved_kyc["id_card"][1], caption="DB ID Card")
            st.info("⬆️ Upload new images to override these saved database documents.")
        else:
            st.info("⬆️ Upload both images for KYC verification.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    btn_col, _ = st.columns([1,3])
    run = btn_col.button("🔍 Run Fraud Analysis", type="primary", use_container_width=False, disabled=not models_ok)

    # ── Analysis ────────────────────────────────────────────
    if run:
        progress_bar = st.progress(0, text="Initialising…")
        status_box   = st.empty()

        try:
            status_box.info("🚀 Preparing data to send to FastAPI Microservice...")
            progress_bar.progress(20, text="Uploading to API...")
            
            # Prepare payload
            data_payload = {
                "amount": str(amount), "type": trans_type,
                "oldbalanceOrg": str(old_balance), "newbalanceOrig": str(new_balance),
                "location": location, "device": device,
                "time_of_day": str(time_of_day), "account_age": str(acc_age),
                "complaint_text": complaint_text.strip() if complaint_text.strip() else "No complaint provided"
            }
            
            # Prepare file uploads
            files_payload = {}
            if selfie_file and id_file:
                # Send fresh override ones
                files_payload["selfie_file"] = (selfie_file.name, selfie_file.getvalue(), "image/jpeg")
                files_payload["id_file"] = (id_file.name, id_file.getvalue(), "image/jpeg")
            elif saved_kyc:
                # Send trusted DB blob
                files_payload["selfie_file"] = (saved_kyc["selfie"][0] or "db_selfie.jpg", saved_kyc["selfie"][1], "image/jpeg")
                files_payload["id_file"] = (saved_kyc["id_card"][0] or "db_id.jpg", saved_kyc["id_card"][1], "image/jpeg")
            
            status_box.info("⚙️ Analyzing transaction, sentiment, and KYC via Microservice...")
            progress_bar.progress(60, text="API Processing...")
            
            response = requests.post(API_URL, data=data_payload, files=files_payload if files_payload else None)
            
            if response.status_code == 200:
                resp_json = response.json()
                t_result = resp_json["t_result"]
                c_result = resp_json["c_result"]
                i_result = resp_json["i_result"]
                fusion_result = resp_json["fusion_result"]
                
                final_score   = fusion_result["final_score"]
                decision      = fusion_result["decision"]
                risk_factors  = fusion_result["risk_factors"]

                # Apply sidebar thresholds
                midpoint = (fraud_threshold + suspicious_threshold) / 2.0
                if final_score >= fraud_threshold:    decision = "FRAUD"
                elif final_score >= midpoint:         decision = "POSSIBLY FRAUD"
                elif final_score >= suspicious_threshold: decision = "POSSIBLY LEGIT"
                else:                                decision = "LEGIT"
                fusion_result["decision"] = decision

                # ── Save to DB ─────────────────────────────────
                selfie_bytes = selfie_file.getvalue() if selfie_file else (saved_kyc["selfie"][1] if saved_kyc else None)
                id_bytes     = id_file.getvalue()     if id_file     else (saved_kyc["id_card"][1] if saved_kyc else None)
                self_name    = selfie_file.name if selfie_file else (saved_kyc["selfie"][0] if saved_kyc else "db_selfie.jpg")
                id_n         = id_file.name     if id_file     else (saved_kyc["id_card"][0] if saved_kyc else "db_id.jpg")
                
                log_id = db.save_analysis(
                    amount=amount, transaction_type=trans_type,
                    old_balance=old_balance, new_balance=new_balance,
                    location=location, device_status=device,
                    time_of_day=time_of_day, account_age=acc_age,
                    complaint_text=complaint_text,
                    t_result=t_result, c_result=c_result, i_result=i_result,
                    fusion_result=fusion_result,
                    selfie_bytes=selfie_bytes, id_bytes=id_bytes,
                    selfie_name=self_name,
                    id_name=id_n,
                    username=username.strip() if username.strip() else "Guest"
                )

                progress_bar.progress(100, text="Done! ✓  Saved to database.")
                time.sleep(0.4)
                progress_bar.empty()
                status_box.success(f"✅ Analysis #{log_id} saved to database.")
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                progress_bar.empty()
                status_box.empty()
                st.stop()

            # ── Results ────────────────────────────────────
            st.markdown("## 📊 Analysis Results")
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            css_class = {"FRAUD":"decision-fraud","POSSIBLY FRAUD":"decision-review","POSSIBLY LEGIT":"decision-legit","LEGIT":"decision-legit"}[decision]
            emoji     = {"FRAUD":"🚨","POSSIBLY FRAUD":"⚠️","POSSIBLY LEGIT":"🔍","LEGIT":"✅"}[decision]
            subtitle  = {
                "FRAUD":  "High-confidence fraud detected. Immediate action recommended.",
                "POSSIBLY FRAUD": "Suspicious activity flagged. High probability of fraud.",
                "POSSIBLY LEGIT": "Minor anomalies detected. Likely legitimate but review cautiously.",
                "LEGIT":  "No significant fraud indicators detected across all vectors.",
            }[decision]

            st.markdown(f"""
            <div class="{css_class}">
                <div class="decision-text">{emoji} {decision}</div>
                <div class="decision-subtitle">{subtitle} &nbsp;·&nbsp; Log ID #{log_id}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")

            g1,g2,g3,g4 = st.columns(4)
            g1.plotly_chart(make_gauge(t_result["score"],       "Transaction Risk",  "#ef4444"), use_container_width=False)
            g2.plotly_chart(make_gauge(c_result["score"],       "Complaint Risk",    "#f59e0b"), use_container_width=False)
            g3.plotly_chart(make_gauge(1-i_result["score"],     "Identity Risk",     "#a78bfa"), use_container_width=False)
            g4.plotly_chart(make_gauge(final_score,             "Fusion Score",      "#60a5fa"), use_container_width=False)

            r1, r2 = st.columns([1,1], gap="large")
            with r1:
                st.markdown('<div class="card"><div class="card-title">🕸️ Multi-Vector Risk Profile</div>', unsafe_allow_html=True)
                st.plotly_chart(make_radar(t_result["score"], c_result["score"], i_result["score"]), use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)

            with r2:
                st.markdown('<div class="card"><div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(make_score_bar({
                    "Transaction Risk": t_result["score"],
                    "Complaint Risk":   c_result["score"],
                    "Identity Risk":    1 - i_result["score"],
                    "Fusion Score":     final_score,
                }), use_container_width=False)
                st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                for label, score in [
                    ("Transaction", t_result["score"]),
                    ("Complaint",   c_result["score"]),
                    ("Identity",    i_result["score"]),
                ]:
                    color = "#ef4444" if score > 0.65 else "#f59e0b" if score > 0.35 else "#22c55e"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:8px 0;border-bottom:1px solid #1e2d40;">
                        <span style="color:#94a3b8;font-size:0.85rem;">{label}</span>
                        <span style="color:{color};font-weight:700;">{score:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;">
                    <span style="color:#e2e8f0;font-weight:600;">Final Fusion Score</span>
                    <span style="color:#60a5fa;font-weight:800;font-size:1.1rem;">{final_score:.4f}</span>
                </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">⚠️ Risk Factors & Evidence</div>', unsafe_allow_html=True)
            if risk_factors and risk_factors[0] != "No significant risk factors identified across analysis vectors":
                rf_cols = st.columns(min(len(risk_factors),2))
                for idx, factor in enumerate(risk_factors):
                    rf_cols[idx%2].markdown(f'<div class="risk-item">🔴 {factor}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-item">✅ No significant risk factors detected.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("🔬 Detailed Model Reasoning"):
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown("**💳 Transaction Model**")
                    for r in t_result.get("reasons",[]): st.markdown(f"- {r}")
                    
                    xai = t_result.get("xai_explanations", {})
                    if xai:
                        st.markdown("<br>**🧠 Explainable AI (XAI) Attributions**", unsafe_allow_html=True)
                        st.caption("PyTorch Input × Gradients calculation")
                        xai_df = pd.DataFrame(list(xai.items()), columns=["Feature", "Impact (%)"])
                        # Sort by absolute impact for cleaner display
                        xai_df["abs_impact"] = xai_df["Impact (%)"].abs()
                        xai_df = xai_df.sort_values(by="abs_impact", ascending=True)
                        
                        fig_xai = px.bar(
                            xai_df, x="Impact (%)", y="Feature", orientation="h",
                            color="Impact (%)", color_continuous_scale=["#fca5a5", "#ef4444"],
                        )
                        fig_xai.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"family":"Inter", "color":"#94a3b8", "size":10},
                            height=180, margin=dict(l=10, r=10, t=10, b=20),
                            coloraxis_showscale=False,
                            xaxis=dict(gridcolor="#1e2d40", title="Decision Impact (%)"),
                            yaxis=dict(title="")
                        )
                        st.plotly_chart(fig_xai, use_container_width=False)
                with d2:
                    st.markdown("**💬 NLP Model**")
                    for r in c_result.get("reasons",[]): st.markdown(f"- {r}")
                with d3:
                    st.markdown("**🪪 KYC/CV Model**")
                    for r in i_result.get("reasons",[]): st.markdown(f"- {r}")

        except Exception as e:
            progress_bar.empty()
            status_box.error(f"❌ Analysis failed: {e}")
            with st.expander("Stack trace"):
                import traceback
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# TAB 2 — Database Logs
# ═══════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("## 🗄️ Stored Analysis Logs")

    filter_col, limit_col, refresh_col = st.columns([2,2,1])
    decision_filter = filter_col.selectbox("Filter by Decision", ["All","FRAUD","REVIEW","LEGIT"], key="df")
    row_limit       = limit_col.slider("Max rows", 10, 500, 100, key="rl")
    if refresh_col.button("🔄 Refresh", use_container_width=False):
        st.rerun()

    logs = db.fetch_logs(limit=row_limit, decision_filter=decision_filter)

    if not logs:
        st.info("No logs found. Run an analysis first.")
    else:
        st.markdown(f"**{len(logs)} record(s) found.**")

        # ── Summary table ───────────────────────────────
        df = pd.DataFrame(logs)
        display_cols = ["id","timestamp","username","decision","final_score","transaction_score",
                        "complaint_score","identity_score","amount","transaction_type",
                        "selfie_uploaded","id_uploaded"]
        df_display = df[[c for c in display_cols if c in df.columns]].copy()
        df_display.rename(columns={
            "id":"#","timestamp":"Time","username":"User","decision":"Decision",
            "final_score":"Fusion","transaction_score":"TX","complaint_score":"NLP",
            "identity_score":"KYC","amount":"Amount","transaction_type":"Type",
            "selfie_uploaded":"Selfie","id_uploaded":"ID Card",
        }, inplace=True)
        for col in ["Fusion","TX","NLP","KYC"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(lambda x: f"{x:.4f}" if x else "—")

        st.dataframe(df_display, use_container_width=False, hide_index=True)

        # ── CSV download ────────────────────────────────
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", data=csv,
            file_name=f"fraud_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # ── Individual row inspector ────────────────────
        st.markdown("### 🔎 Inspect a Record")
        log_ids = [r["id"] for r in logs]
        selected_id = st.selectbox("Select Log ID", log_ids, key="sel_id")

        if selected_id:
            record = next((r for r in logs if r["id"] == selected_id), None)
            if record:
                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                d_color = "#ef4444" if record["decision"]=="FRAUD" else "#f59e0b" if record["decision"]=="REVIEW" else "#22c55e"
                ic1.markdown(f"**Decision:** <span style='color:{d_color};font-weight:700'>{record['decision']}</span>", unsafe_allow_html=True)
                ic2.markdown(f"**User:** {record.get('username', 'Guest')}")
                ic3.metric("Fusion Score",      f"{record['final_score']:.4f}")
                ic4.metric("TX Score",          f"{record['transaction_score']:.4f}")
                ic5.metric("KYC Score",         f"{record['identity_score']:.4f}")

                st.markdown(f"**Complaint:** *{record['complaint_text']}*")

                reasons_col1, reasons_col2 = st.columns(2)
                with reasons_col1:
                    st.markdown("**Risk Factors**")
                    for r in json.loads(record.get("risk_factors") or "[]"):
                        st.markdown(f'<div class="risk-item">🔴 {r}</div>', unsafe_allow_html=True)
                with reasons_col2:
                    st.markdown("**Transaction Reasons**")
                    for r in json.loads(record.get("transaction_reasons") or "[]"):
                        st.markdown(f"- {r}")
                    st.markdown("**NLP Reasons**")
                    for r in json.loads(record.get("complaint_reasons") or "[]"):
                        st.markdown(f"- {r}")

                # Show stored images if any
                uploads = db.fetch_uploads_for(selected_id)
                if uploads:
                    st.markdown("**📷 Stored Images**")
                    img_cols = st.columns(len(uploads))
                    for idx, u in enumerate(uploads):
                        img = Image.open(BytesIO(u["file_data"]))
                        img_cols[idx].image(img, caption=f"{u['file_type']} — {u['filename']}")

                # Delete button
                if st.button(f"🗑️ Delete Log #{selected_id}", key="del_btn"):
                    db.delete_log(selected_id)
                    st.success(f"Log #{selected_id} deleted.")
                    st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ═══════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("## 📈 Database Analytics")

    all_logs = db.fetch_logs(limit=1000)
    if not all_logs:
        st.info("No data yet. Run some analyses first.")
    else:
        df_all = pd.DataFrame(all_logs)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        df_all = df_all.sort_values("timestamp")

        # ── Decision pie ────────────────────────────────
        ac1, ac2 = st.columns(2)
        with ac1:
            counts = df_all["decision"].value_counts().reset_index()
            counts.columns = ["Decision", "Count"]
            fig_pie = px.pie(
                counts, names="Decision", values="Count",
                color="Decision",
                color_discrete_map={"FRAUD":"#ef4444","POSSIBLY FRAUD":"#f59e0b","POSSIBLY LEGIT":"#fcd34d","LEGIT":"#22c55e"},
                hole=0.55,
                title="Decision Distribution",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family":"Inter","color":"#94a3b8"},
                title_font_color="#e2e8f0",
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_pie, use_container_width=False)

        # ── Score distribution ────────────────────────
        with ac2:
            fig_hist = px.histogram(
                df_all, x="final_score", nbins=20,
                color="decision",
                color_discrete_map={"FRAUD":"#ef4444","POSSIBLY FRAUD":"#f59e0b","POSSIBLY LEGIT":"#fcd34d","LEGIT":"#22c55e"},
                title="Fusion Score Distribution",
                labels={"final_score":"Fusion Score","count":"Count"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family":"Inter","color":"#94a3b8"},
                title_font_color="#e2e8f0",
                xaxis=dict(gridcolor="#1e2d40"),
                yaxis=dict(gridcolor="#1e2d40"),
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_hist, use_container_width=False)

        # ── Score timeline ────────────────────────────
        fig_line = px.line(
            df_all, x="timestamp", y="final_score",
            color="decision",
            color_discrete_map={"FRAUD":"#ef4444","POSSIBLY FRAUD":"#f59e0b","POSSIBLY LEGIT":"#fcd34d","LEGIT":"#22c55e"},
            title="Fusion Score Over Time",
            markers=True,
            labels={"timestamp":"Time","final_score":"Fusion Score"},
        )
        fig_line.add_hline(y=0.75, line_dash="dash", line_color="#ef4444", annotation_text="Fraud threshold")
        fig_line.add_hline(y=0.40, line_dash="dash", line_color="#f59e0b", annotation_text="Review threshold")
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family":"Inter","color":"#94a3b8"},
            title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2d40"),
            yaxis=dict(gridcolor="#1e2d40"),
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_line, use_container_width=False)

        # ── Model avg scores bar ───────────────────────
        avg_data = {
            "Model": ["Transaction", "Complaint", "Identity"],
            "Avg Score": [
                stats.get("avg_tx") or 0,
                stats.get("avg_nlp") or 0,
                stats.get("avg_kyc") or 0,
            ]
        }
        fig_avg = px.bar(
            avg_data, x="Model", y="Avg Score",
            color="Avg Score",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            title="Average Score by Model",
            range_y=[0, 1],
        )
        fig_avg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family":"Inter","color":"#94a3b8"},
            title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2d40"),
            yaxis=dict(gridcolor="#1e2d40"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_avg, use_container_width=False)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:8px 0;">
    🛡️ AI Fraud Detection System &nbsp;·&nbsp; Deep Learning · NLP · Computer Vision · Fusion Engine
    &nbsp;·&nbsp; Powered by SQLite &nbsp;·&nbsp; v5.0
</div>
""", unsafe_allow_html=True)
