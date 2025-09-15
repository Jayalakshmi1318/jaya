import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('transactions.csv')

# Encode card_number (simple hash for demo)
df['card_number'] = df['card_number'].apply(lambda x: hash(x) % 10**6)

# Features and target
X = df[['card_number', 'amount']]
y = df['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fraud_model.pkl')
print("✅ Model trained and saved as fraud_model.pkl")