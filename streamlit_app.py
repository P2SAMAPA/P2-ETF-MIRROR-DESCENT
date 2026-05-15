import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Mirror Descent Online Portfolio", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-weight { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🪞 Mirror Descent Online Portfolio Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Exponential Gradient (EG) update | KL mirror descent | Online convex optimisation</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🪞 Mirror Descent")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Learning rate (η):** {config.LEARNING_RATE}")
st.sidebar.markdown(f"**Adaptive:** {'Yes' if config.USE_ADAPTIVE else 'No'}")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'mirror_descent_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Current Portfolio Weights (Top Holdings)")

for universe_name, uni_data in universes.items():
    top_assets = uni_data.get("top_assets", [])
    if not top_assets:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, asset in enumerate(top_assets):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{asset['ticker']}</div>
                <div class="etf-weight">weight = {asset['weight']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Cumulative Wealth", f"{uni_data.get('final_cum_wealth', 0):.3f}", help="Starting at 1.0")
    with col2:
        st.metric("Last Portfolio Return", f"{uni_data.get('last_portfolio_return', 0):.4f}")
    with st.expander("📋 Full weight vector (all assets)"):
        full = uni_data.get("full_weights", {})
        if full:
            df = pd.DataFrame(list(full.items()), columns=["ETF", "Weight"])
            df = df.sort_values("Weight", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Exponential Gradient (EG) update: w_{t+1,i} = w_{t,i} * exp(η * r_{t,i}) / Z. This is mirror descent with KL divergence. The portfolio is rebalanced daily after observing returns.")
