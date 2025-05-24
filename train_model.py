import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data load karo
df = pd.read_csv("spacex_launch_data.csv")

# Clean karna: sirf rows jisme 'success' value available ho
df = df[df['success'].notnull()]

# Encode success (True/False) to 1/0
df['success'] = df['success'].astype(int)

# Filhaal sirf launchpad aur payloads ka count use karein (simple model)
df['payload_count'] = df['payloads'].apply(lambda x: len(str(x).split(',')))
X = df[['payload_count']]  # Input feature
y = df['success']          # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
