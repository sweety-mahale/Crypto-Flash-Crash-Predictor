import os
import time
import requests
import streamlit as pd_st
import pandas as pd
import plotly.graph_objects as go
import redis

# Custom styling for rich aesthetics
pd_st.set_page_config(page_title="Crypto Crash Live Monitor", layout="wide")

pd_st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .metric-box {
        background-color: #1a1e29;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
</style>
""", unsafe_allow_html=True)

pd_st.title("⚡ Live Cryptocurrency Flash Crash Predictor Dashboard")
pd_st.subheader("Production Online ML System updating via Binance Websockets")

api_url = os.getenv("API_URL", "http://localhost:8000")
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# Metrics boxes placeholder
metrics_placeholder = pd_st.empty()
chart_placeholder = pd_st.empty()

# We maintain local cache for charts plotting
if 'price_history' not in pd_st.session_state:
    pd_st.session_state['price_history'] = []
if 'risk_history' not in pd_st.session_state:
    pd_st.session_state['risk_history'] = []
if 'time_history' not in pd_st.session_state:
    pd_st.session_state['time_history'] = []

while True:
    try:
        res = requests.get(f"{api_url}/metrics", timeout=2).json()
        latest = res.get("latest_features", {})
        samples = res.get("samples_trained", 0)
        drift = res.get("drift_detected", False)
        
        if latest:
            current_price = float(latest.get("price", 0.0))
            current_risk = float(latest.get("risk", 0.0))
            
            # Update history
            pd_st.session_state['price_history'].append(current_price)
            pd_st.session_state['risk_history'].append(current_risk)
            pd_st.session_state['time_history'].append(time.strftime("%H:%M:%S"))
            
            # Keep history to last 50 ticks
            if len(pd_st.session_state['price_history']) > 50:
                pd_st.session_state['price_history'].pop(0)
                pd_st.session_state['risk_history'].pop(0)
                pd_st.session_state['time_history'].pop(0)
                
            # Render metric boxes
            with metrics_placeholder.container():
                col1, col2, col3, col4 = pd_st.columns(4)
                with col1:
                    pd_st.metric("BTC price", f"${current_price:,.2f}")
                with col2:
                    pd_st.metric("Model Crash Risk", f"{current_risk*100:.1f}%")
                with col3:
                    pd_st.metric("Ticks Trained", f"{samples:,}")
                with col4:
                    status = "⚠️ DRIFT DETECTED" if drift else "🟢 HEALTHY"
                    pd_st.metric("Concept Drift Status", status)
                    
            # Plot charts
            with chart_placeholder.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd_st.session_state['time_history'], 
                    y=pd_st.session_state['price_history'],
                    mode='lines+markers',
                    name='BTC Price',
                    line=dict(color='#00ffcc', width=2)
                ))
                fig.update_layout(
                    title="Real-Time Price & Volatility Tracker",
                    xaxis_title="Time",
                    yaxis_title="Price (USDT)",
                    template="plotly_dark",
                    height=450
                )
                pd_st.plotly_chart(fig, use_container_width=True)
                
                # Bottom risk charts
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Bar(
                    x=pd_st.session_state['time_history'],
                    y=pd_st.session_state['risk_history'],
                    name='Crash Risk Probability',
                    marker_color='#ff3366'
                ))
                fig_risk.update_layout(
                    title="Model Calculated Flash Crash Risk Profile (Next 5 min)",
                    xaxis_title="Time",
                    yaxis_title="Probability",
                    template="plotly_dark",
                    height=300
                )
                pd_st.plotly_chart(fig_risk, use_container_width=True)
                
        else:
            pd_st.info("Waiting for tick messages to flow through Redis Stream...")
            
    except Exception as e:
        pd_st.error(f"Error fetching real-time metrics: {e}")
        
    time.sleep(1)
