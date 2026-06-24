import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("dataset.csv")

X = data[["Age", "Sleep_Hours", "Work_Hours"]]
y = data["Stress_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)
with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))
joblib.dump(model, "model.pkl")