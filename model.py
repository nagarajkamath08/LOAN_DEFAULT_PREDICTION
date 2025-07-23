import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load dataset
df = pd.read_csv('cleaned_loan_data.csv')

# Drop unnecessary columns
df = df.drop(columns=[col for col in df.columns if col.lower() in ['issue_d', 'desc', 'title', 'id']], errors='ignore')

# Define target and features
if 'Default' not in df.columns:
    raise ValueError("Target column 'Default' not found in dataset.")

X = df.drop('Default', axis=1)
y = df['Default']

# Handle categorical variables safely
categorical_cols = X.select_dtypes(include='object').columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on full unique values to allow unseen values to be mapped later
    unique_values = X[col].dropna().unique().tolist()
    le.fit(unique_values)
    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])  # fallback to first
    X[col] = le.transform(X[col])
    encoders[col] = le

# Save column order for consistent frontend input
feature_order = list(X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# Save all components
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(encoders, open("model/encoders.pkl", "wb"))
pickle.dump(feature_order, open("model/feature_order.pkl", "wb"))

print("✅ Model trained and saved successfully.")
