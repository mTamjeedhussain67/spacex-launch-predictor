import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("spacex_launch_data.csv")


print("🧾 Data Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())
print("🧪 Null values:\n", df.isnull().sum())


print("🚀 Success/Failure Count:")
print(df['success'].value_counts())


sns.countplot(data=df, x='success')
plt.title("Success vs Failure")
plt.show(block=True)



