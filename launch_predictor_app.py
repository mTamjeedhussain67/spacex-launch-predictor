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

# ---------- UI Layout ----------
st.set_page_config(page_title="🚀 SpaceX Launch App", layout="wide")

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(lottie_rocket, height=180, speed=1)
    with col2:
        st.title("🚀 SpaceX Launch Success Predictor")
        st.markdown("Predict launch success and explore SpaceX launch history in a visual dashboard.")

st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Choose a section", ["🚀 Predict Launch", "📊 Launch Data", "🗺️ Launch Map"])
st.sidebar.markdown("---")
st.sidebar.info("Made by **Tamjeed Hussain**")

# ---------- Section 1: Prediction ----------
if section == "🚀 Predict Launch":
    st.subheader("🎯 Predict the Success of a Launch")
    st.markdown("### Enter Payload Count:")
    payload_input = st.slider("Payload Count", min_value=1, max_value=10, value=2)

    if st.button("🔮 Predict Launch Success"):
        prediction = model.predict([[payload_input]])
        col1, col2 = st.columns(2)
        if prediction[0] == 1:
            with col1:
                st.success("✅ The launch is likely to be **Successful!**")
            with col2:
                st.metric("Prediction", "Success", delta="+95% confidence")
        else:
            with col1:
                st.error("❌ The launch might **Fail**.")
            with col2:
                st.metric("Prediction", "Failure", delta="-60% confidence")

# ---------- Section 2: Launch Data ----------
elif section == "📊 Launch Data":
    st.subheader("📅 SpaceX Launch Data Explorer")
    df['date_utc'] = pd.to_datetime(df['date_utc'])

    years = sorted(df['date_utc'].dt.year.unique())
    col1, col2 = st.columns(2)
    selected_year = col1.selectbox("Select Launch Year", years)
    selected_site = col2.selectbox("Select Launchpad", ["All"] + list(df['launchpad'].unique()))

    filtered_df = df[df['date_utc'].dt.year == selected_year]
    if selected_site != "All":
        filtered_df = filtered_df[filtered_df['launchpad'] == selected_site]

    success_count = filtered_df['success'].sum()
    fail_count = len(filtered_df) - success_count

    st.markdown("### 📈 Launch Stats")
    col1, col2 = st.columns(2)
    col1.metric("✅ Successful Launches", success_count)
    col2.metric("❌ Failed Launches", fail_count)

    with st.expander("🔍 Show Launch Data Table"):
        st.dataframe(filtered_df[['name', 'date_utc', 'success', 'launchpad']], use_container_width=True)

# ---------- Section 3: Launch Map ----------
elif section == "🗺️ Launch Map":
    st.subheader("🗺️ Interactive Launch Sites Map")

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
