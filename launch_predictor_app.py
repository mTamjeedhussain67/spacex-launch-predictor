import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier

# --- Page Config ---
st.set_page_config(page_title="SpaceX Launch Predictor", layout="wide")

# --- Title ---
st.title("🚀 SpaceX Launch Analysis & Success Predictor")
st.caption("Powered by Python, Machine Learning & Streamlit UI")

# --- Load & Train Model ---
@st.cache_resource
def train_model():
    df = pd.read_csv("spacex_launch_data.csv")
    df = df[df['success'].notnull()]
    df['success'] = df['success'].astype(int)
    df['payload_count'] = df['payloads'].apply(lambda x: len(str(x).split(',')))
    X = df[['payload_count']]
    y = df['success']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, df

model, df = train_model()

# --- Sidebar Inputs ---
st.sidebar.header("🧪 Predict Launch Outcome")
payload_input = st.sidebar.slider("Payload Count", 1, 10, 2)
predict_button = st.sidebar.button("Predict Launch Success")

# --- Tabbed Layout ---
tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "📊 Launch Data", "🗺 Launch Map"])

# --- Prediction Tab ---
with tab1:
    st.subheader("🎯 Launch Success Prediction")
    if predict_button:
        prediction = model.predict([[payload_input]])
        if prediction[0] == 1:
            st.success("✅ The Launch is Likely to be SUCCESSFUL!")
        else:
            st.error("❌ The Launch Might FAIL.")
    else:
        st.info("Adjust the payload count and click **Predict Launch Success**.")

# --- Data Tab ---
with tab2:
    st.subheader("📊 Historical Launch Data")
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    years = sorted(df['date_utc'].dt.year.unique())
    selected_year = st.selectbox("Filter by Year", years, index=len(years)-1)

    filtered_df = df[df['date_utc'].dt.year == selected_year]
    st.dataframe(
        filtered_df[['name', 'date_utc', 'success', 'launchpad']],
        use_container_width=True
    )

# --- Map Tab ---
with tab3:
    st.subheader("🗺 Launchpad Locations Map")

    launchpad_coords = {
        '5e9e4502f5090995de566f86': (28.5623, -80.5774),  # CCAFS SLC 40
        '5e9e4501f509094ba4566f84': (34.6321, -120.6106), # VAFB SLC 4E
        '5e9e4502f509092b78566f87': (28.4858, -80.5449),  # KSC LC 39A
        '5e9e4502f509094188566f88': (25.9972, 97.3546)    # Starbase Boca Chica
    }

    m = folium.Map(location=[28.5, -80.6], zoom_start=4)

    for idx, row in df.iterrows():
        coords = launchpad_coords.get(row['launchpad'], None)
        if coords:
            status = "✅ Success" if row['success'] == 1 else "❌ Failure"
            popup = f"<b>{row['name']}</b><br>{row['date_utc'].strftime('%Y-%m-%d')}<br>Status: {status}"
            color = "green" if row['success'] == 1 else "red"
            folium.Marker(
                location=coords,
                popup=popup,
                icon=folium.Icon(color=color)
            ).add_to(m)

    st_data = st_folium(m, width=700, height=500)
