import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

# Save model
pickle.dump(model, open("diabetes_model.sav", "wb"))

print("Model trained and saved successfully")