import requests
import pandas as pd


url = "https://api.spacexdata.com/v4/launches"


response = requests.get(url)
data = response.json()


df = pd.json_normalize(data)


columns_needed = ['name', 'date_utc', 'success', 'rocket', 'payloads', 'launchpad']
df = df[columns_needed]


df.to_csv("spacex_launch_data.csv", index=False)

print("✅ SpaceX launch data downloaded and saved as 'spacex_launch_data.csv'")
