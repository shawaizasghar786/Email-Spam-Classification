import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the original dataset to get column names (features)
data = pd.read_csv("spam.csv")
feature_columns = data.select_dtypes(include=['number']).drop(columns=['Prediction'], errors='ignore').columns

# Function to convert raw text into feature vector
def text_to_features(text):
    words = text.lower().split()
    features = np.zeros(len(feature_columns))
    for idx, word in enumerate(feature_columns):
        features[idx] = words.count(word)
    return [features]

# Take user input
user_text = input("✍️ Enter your email content:\n")

# Convert to features
sample_input = text_to_features(user_text)

# Predict
prediction = model.predict(sample_input)[0]
print("\n🔎 Prediction:", "Spam 🚨" if prediction == 1 else "Not Spam ✅")
