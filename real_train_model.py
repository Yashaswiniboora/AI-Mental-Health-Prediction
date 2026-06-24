import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("student_mental_health_burnout.csv")

# Convert stress levels to numbers
mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

data["stress_level"] = data["stress_level"].map(mapping)

# Features
X = data[[
    "age",
    "daily_sleep_hours",
    "anxiety_score",
    "depression_score",
    "academic_pressure_score",
    "financial_stress_score",
    "social_support_score",
    "physical_activity_hours",
    "attendance_percentage",
    "cgpa"
]]

# Target
y = data["stress_level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)

# Save model
joblib.dump(model, "real_model.pkl")

print("Real Model Saved Successfully!")