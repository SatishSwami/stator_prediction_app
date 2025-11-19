# app.py - Premium Glass UI + Motor Background + CSV Export Only

import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(page_title="Stator Winding Temperature Predictor",
                   page_icon="🔧", layout="wide")

BASE = Path.cwd()
MODEL_PATH = BASE / "model.pkl"
PRE_PATH = BASE / "preprocessor.pkl"
BG_PATH = BASE / "bg.jpg"
LOGO_PATH = BASE / "mit_logo.jpg"
CSS_PATH = BASE / "styles.css"

# ---------------- Load CSS ----------------
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- Background Image ----------------
if BG_PATH.exists():
    st.markdown(
        f"""
        <style>
        .stApp {{
          background-image: linear-gradient(rgba(255,255,255,0.60), rgba(255,255,255,0.60)), url("{BG_PATH.name}");
          background-size: cover;
          background-position: center;
          background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Load Model & Preprocessor ----------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PRE_PATH)
    return model, pre

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model/preprocessor: {e}")
    st.stop()

# ---------------- Header ----------------
col1, col2 = st.columns([1, 6])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
with col2:
    st.markdown("<div class='header-title'>🔧 Stator Winding Temperature Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Premium Glass UI • MITAOE Project</div>", unsafe_allow_html=True)

st.write("")

# ---------------- Layout ----------------
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Input Measurements")

    presets = st.selectbox("Presets", ["Custom", "Normal run", "High load", "Low speed"])

    if presets == "Normal run":
        defaults = dict(stator_tooth=50, stator_yoke=40, coolant=25, motor_speed=1500, torque=10)
    elif presets == "High load":
        defaults = dict(stator_tooth=65, stator_yoke=55, coolant=30, motor_speed=2000, torque=25)
    elif presets == "Low speed":
        defaults = dict(stator_tooth=40, stator_yoke=35, coolant=20, motor_speed=800, torque=5)
    else:
        defaults = dict(stator_tooth=50, stator_yoke=40, coolant=25, motor_speed=1500, torque=10)

    stator_tooth = st.number_input("Stator Tooth", value=float(defaults["stator_tooth"]))
    stator_yoke  = st.number_input("Stator Yoke", value=float(defaults["stator_yoke"]))
    coolant      = st.number_input("Coolant (°C)", value=float(defaults["coolant"]))
    motor_speed  = st.number_input("Motor Speed (RPM)", value=float(defaults["motor_speed"]))
    torque       = st.number_input("Torque (Nm)", value=float(defaults["torque"]))

    st.write("")
    predict_button = st.button("Predict", help="Run ML Model")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Prediction")
    result_box = st.empty()
    therm_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Safe Transform ----------------
def prepare_input(pre, X):
    try:
        return pre.transform(X)
    except:
        needed = pre.n_features_in_
        current = X.shape[1]
        if current < needed:
            X = np.hstack([X, np.zeros((1, needed - current))])
        return pre.transform(X)

# ---------------- Run Prediction ----------------
if predict_button:
    try:
        X_raw = np.array([[stator_tooth, stator_yoke, coolant, motor_speed, torque]])
        Xp = prepare_input(preprocessor, X_raw)
        pred = float(model.predict(Xp)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Show result
    result_box.markdown(f"<div class='result-big'>{pred:.2f} °C</div>", unsafe_allow_html=True)

    # Vertical thermometer gauge
    max_val = max(120, pred * 1.6)
    df = pd.DataFrame({"y": [pred]})

    therm = alt.Chart(df).mark_bar(size=70).encode(
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, max_val])),
        x=alt.value(40),
        color=alt.Color("y:Q", scale=alt.Scale(domain=[0, max_val], scheme="orangered")),
    ).properties(width=120, height=380)

    line = alt.Chart(pd.DataFrame({"y": [pred]})).mark_rule(
        color="white", strokeWidth=3
    ).encode(y="y:Q")

    therm_box.altair_chart(therm + line, use_container_width=False)

    # Save to history
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stator_tooth": stator_tooth,
        "stator_yoke": stator_yoke,
        "coolant": coolant,
        "motor_speed": motor_speed,
        "torque": torque,
        "prediction": round(pred, 3)
    }

    st.session_state.setdefault("history", []).insert(0, entry)

# ---------------- History & CSV Export ----------------
if st.session_state.get("history"):
    st.subheader("Recent Predictions")
    df_hist = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_hist.head(10), use_container_width=True)

    csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv")

# ---------------- Footer ----------------
st.markdown("<div class='app-footer'>MITAOE • ML Model Deployment Demo</div>", unsafe_allow_html=True)
