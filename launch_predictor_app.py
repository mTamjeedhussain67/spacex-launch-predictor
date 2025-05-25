import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import requests

# ---------- Helper Functions ----------
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

# ---------- Load Resources ----------
df, model = load_data_and_model()
lottie_rocket = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_ig8fvpyk.json")

# ---------- Page Config ----------
st.set_page_config(page_title="🚀 SpaceX Launch Predictor", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            color: #58a6ff;
        }
        .section-header {
            font-size: 1.8rem;
            margin-top: 30px;
            color: #58a6ff;
            border-left: 4px solid #58a6ff;
            padding-left: 10px;
        }
        .card {
            background-color: #161b22;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .metric {
            font-size: 1.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header Section ----------
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st_lottie(lottie_rocket, height=160, speed=1)
    with col2:
        st.markdown("<div class='main-title'>🚀 SpaceX Launch Success Predictor</div>", unsafe_allow_html=True)
        st.markdown("Get instant predictions, browse historical data, and explore launch locations in a sleek, modern dashboard.")

# ---------- Sidebar ----------
st.sidebar.title("🔍 Menu")
section = st.sidebar.radio("Navigate to:", ["🚀 Predict", "📊 Data", "🗺️ Map"])
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ by **Tamjeed Hussain**")

# ---------- Prediction Section ----------
if section == "🚀 Predict":
    st.markdown("<div class='section-header'>🎯 Predict Launch Outcome</div>", unsafe_allow_html=True)
    with st.container():
        with st.form("predict_form"):
            payload_input = st.slider("Payload Count", 1, 10, 2)
            submitted = st.form_submit_button("🔮 Predict Now")
            if submitted:
                prediction = model.predict([[payload_input]])
                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        st.success("✅ Launch likely to be **Successful**!")
                    else:
                        st.error("❌ Launch may **Fail**.")
                with col2:
                    st.metric("Prediction Result", "Success" if prediction[0] else "Failure")

# ---------- Launch Data Section ----------
elif section == "📊 Data":
    st.markdown("<div class='section-header'>📅 Launch Data Explorer</div>", unsafe_allow_html=True)
    df['date_utc'] = pd.to_datetime(df['date_utc'])

    with st.container():
        years = sorted(df['date_utc'].dt.year.unique())
        col1, col2 = st.columns(2)
        selected_year = col1.selectbox("Choose Year", years)
        selected_site = col2.selectbox("Choose Launch Site", ["All"] + list(df['launchpad'].unique()))

        filtered_df = df[df['date_utc'].dt.year == selected_year]
        if selected_site != "All":
            filtered_df = filtered_df[filtered_df['launchpad'] == selected_site]

        success_count = filtered_df['success'].sum()
        fail_count = len(filtered_df) - success_count

        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        col1.metric("✅ Success", success_count)
        col2.metric("❌ Failure", fail_count)

        with st.expander("📋 Show Data Table"):
            st.dataframe(filtered_df[['name', 'date_utc', 'success', 'launchpad']], use_container_width=True)

# ---------- Launch Map Section ----------
elif section == "🗺️ Map":
    st.markdown("<div class='section-header'>🗺️ Launch Sites Map</div>", unsafe_allow_html=True)

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
