import requests
import pandas as pd

# SpaceX API endpoint
url = "https://api.spacexdata.com/v4/launches"

# API se data fetch karna
response = requests.get(url)
data = response.json()

# JSON data ko DataFrame mein convert karna
df = pd.json_normalize(data)

# Sirf kuch useful columns select karna
columns_needed = ['name', 'date_utc', 'success', 'rocket', 'payloads', 'launchpad']
df = df[columns_needed]

# CSV file mein save karna
df.to_csv("spacex_launch_data.csv", index=False)

print("✅ SpaceX launch data downloaded and saved as 'spacex_launch_data.csv'")
