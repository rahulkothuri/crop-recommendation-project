import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("data\Crop_recommendation.csv")

# Splitting features and target
X = data.drop(columns=["label"])  # Features
y = data["label"]  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
