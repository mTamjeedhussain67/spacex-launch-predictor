import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import requests

# Load Data and Model
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("spacex_launch_data.csv")
    df = df[df['success'].notnull()]
    df['success'] = df['success'].astype(int)
    df['payload_count'] = df['payloads'].apply(lambda x: len(str(x).split(',')))

    X = df[['payload_count']]
    y = df['success']

    model = RandomForestClassifier()
    model.fit(X, y)
    return df, model

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load
df, model = load_data_and_model()
lottie_rocket = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_ig8fvpyk.json")

# Page Setup
st.set_page_config(page_title="🚀 SpaceX Launch Predictor", layout="wide")

# ===== Custom Stylish CSS =====
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-top: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
    }
    .title-text {
        font-size: 3em;
        font-weight: bold;
        color: #58a6ff;
    }
    .sub-heading {
        font-size: 1.4em;
        color: #8b949e;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 2em;
        font-weight: 600;
        border-left: 5px solid #58a6ff;
        padding-left: 10px;
        margin-top: 50px;
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Header with Animation =====
col1, col2 = st.columns([1, 3])
with col1:
    st_lottie(lottie_rocket, height=150)
with col2:
    st.markdown("<div class='title-text'>🚀 SpaceX Launch Success Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-heading'>Smart prediction engine + clean dashboard UI = mission control at your fingertips.</div>", unsafe_allow_html=True)

# ===== Sidebar Navigation =====
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/SpaceX-Logo.svg/320px-SpaceX-Logo.svg.png", width=180)
section = st.sidebar.radio("🔍 Navigate", ["🎯 Predict", "📊 Launch Data", "🗺️ Launch Map"])
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ by Tamjeed Hussain")

# ===== Section 1: Prediction =====
if section == "🎯 Predict":
    st.markdown("<div class='section-title'>🎯 Predict Launch Success</div>", unsafe_allow_html=True)
    with st.container():
        with st.form("predict_form"):
            st.markdown("<div class='main-box'>", unsafe_allow_html=True)
            payload = st.slider("Select Payload Count", 1, 10, 3)
            submitted = st.form_submit_button("🚀 Predict")
            st.markdown("</div>", unsafe_allow_html=True)

            if submitted:
                prediction = model.predict([[payload]])
                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        st.success("✅ Likely to be **Successful**!")
                    else:
                        st.error("❌ Might **Fail**.")
                with col2:
                    st.metric("Confidence Level", "High" if prediction[0] else "Low")

# ===== Section 2: Launch Data =====
elif section == "📊 Launch Data":
    st.markdown("<div class='section-title'>📊 SpaceX Launch Records</div>", unsafe_allow_html=True)
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    years = sorted(df['date_utc'].dt.year.unique())

    col1, col2 = st.columns(2)
    year = col1.selectbox("Select Year", years)
    site = col2.selectbox("Select Launchpad", ["All"] + list(df['launchpad'].unique()))

    filtered_df = df[df['date_utc'].dt.year == year]
    if site != "All":
        filtered_df = filtered_df[filtered_df['launchpad'] == site]

    success = filtered_df['success'].sum()
    failure = len(filtered_df) - success

    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Successful", success)
    with col2:
        st.metric("❌ Failed", failure)

    with st.expander("📋 Show Full Launch Table"):
        st.dataframe(filtered_df[['name', 'date_utc', 'success', 'launchpad']], use_container_width=True)

# ===== Section 3: Launch Map =====
elif section == "🗺️ Launch Map":
    st.markdown("<div class='section-title'>🗺️ Launch Sites Map</div>", unsafe_allow_html=True)

    launchpad_coords = {
        '5e9e4502f5090995de566f86': (28.5623, -80.5774),  # CCAFS SLC 40
        '5e9e4501f509094ba4566f84': (34.6321, -120.6106), # VAFB SLC 4E
        '5e9e4502f509092b78566f87': (28.4858, -80.5449),  # KSC LC 39A
        '5e9e4502f509094188566f88': (25.9972, 97.3546)    # Starbase Boca Chica
    }

    m = folium.Map(location=[28.5, -80.6], zoom_start=4)
    for idx, row in df.iterrows():
        coords = launchpad_coords.get(row['launchpad'])
        if coords:
            status = "✅ Success" if row['success'] == 1 else "❌ Failure"
            popup = f"{row['name']}<br>{row['date_utc'].strftime('%Y-%m-%d')}<br>Status: {status}"
            color = "green" if row['success'] == 1 else "red"
            folium.Marker(location=coords, popup=popup, icon=folium.Icon(color=color)).add_to(m)

    st_folium(m, width=900, height=500)
