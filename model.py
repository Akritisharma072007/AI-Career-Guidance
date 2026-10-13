import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting AI Career Prediction Model...")

# Load dataset
data = pd.read_csv("career_dataset.csv", header=None)

print("Dataset loaded successfully!")
print("Number of records:", len(data))

# Combine Skills + Interests + Favourite Subject
feature_data = data[[0, 1, 2, 3, 4, 5]].fillna("")

data["combined_features"] = feature_data.apply(
    lambda row: ",".join(row.astype(str)),
    axis=1
)

# Convert features into numerical form
mlb = MultiLabelBinarizer()

X = mlb.fit_transform(
    data["combined_features"].apply(
        lambda x: [item.strip() for item in x.split(",")]
    )
)

# Career column
y = data[6]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

print("Model trained successfully!")

# Save model and encoder
joblib.dump(model, "career_model.pkl")
joblib.dump(mlb, "skill_encoder.pkl")

print("AI Career Prediction Model Ready!")
print("Model saved successfully.")
