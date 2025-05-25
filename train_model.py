import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("spacex_launch_data.csv")


df = df[df['success'].notnull()]


df['success'] = df['success'].astype(int)


df['payload_count'] = df['payloads'].apply(lambda x: len(str(x).split(',')))
X = df[['payload_count']]  
y = df['success']          


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
