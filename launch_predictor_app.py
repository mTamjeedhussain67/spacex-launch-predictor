import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Train model
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
    return model

model = train_model()

# Streamlit UI
st.title("🚀 SpaceX Launch Success Predictor")

payload_input = st.slider("Enter Payload Count", min_value=1, max_value=10, value=2)

if st.button("Predict"):
    prediction = model.predict([[payload_input]])
    if prediction[0] == 1:
        st.success("✅ Launch Likely to be SUCCESSFUL!")
    else:
        st.error("❌ Launch Might FAIL.")

st.markdown("---")
st.header("📚 Historical SpaceX Launch Data")

df = pd.read_csv("spacex_launch_data.csv")

# Convert to datetime
df['date_utc'] = pd.to_datetime(df['date_utc'])

# Sidebar filters
years = sorted(df['date_utc'].dt.year.unique())
selected_year = st.selectbox("Select Launch Year", years)

# Filter by year
filtered_df = df[df['date_utc'].dt.year == selected_year]

# Optional: Launch site filter (basic)
# launch_sites = df['launchpad'].unique()
# selected_site = st.selectbox("Select Launchpad", launch_sites)
# filtered_df = filtered_df[filtered_df['launchpad'] == selected_site]

st.dataframe(filtered_df[['name', 'date_utc', 'success', 'launchpad']])

import folium
from streamlit_folium import st_folium

st.markdown("---")
st.header("🗺️ Launch Sites Map (Simple View)")

# Sample coordinates for launchpads (static for now)
launchpad_coords = {
    '5e9e4502f5090995de566f86': (28.5623, -80.5774),  # CCAFS SLC 40
    '5e9e4501f509094ba4566f84': (34.6321, -120.6106), # VAFB SLC 4E
    '5e9e4502f509092b78566f87': (28.4858, -80.5449),  # KSC LC 39A
    '5e9e4502f509094188566f88': (25.9972, 97.3546)    # Starbase Boca Chica (example)
}

# Create a folium map
m = folium.Map(location=[28.5, -80.6], zoom_start=4)

for idx, row in df.iterrows():
    launchpad = row['launchpad']
    coords = launchpad_coords.get(launchpad, None)
    if coords:
        status = "✅ Success" if row['success'] == 1 else "❌ Failure"
        popup = f"{row['name']} ({row['date_utc'].strftime('%Y-%m-%d')})<br>Status: {status}"
        color = "green" if row['success'] == 1 else "red"
        folium.Marker(
            location=coords,
            popup=popup,
            icon=folium.Icon(color=color)
        ).add_to(m)

# Show folium map in Streamlit
st_data = st_folium(m, width=700, height=500)
