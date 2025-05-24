import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV file load karo
df = pd.read_csv("spacex_launch_data.csv")

# Basic info
print("🧾 Data Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())
print("🧪 Null values:\n", df.isnull().sum())

# Success/Failure count
print("🚀 Success/Failure Count:")
print(df['success'].value_counts())

# Plot success/failure
sns.countplot(data=df, x='success')
plt.title("Success vs Failure")
plt.show(block=True)



