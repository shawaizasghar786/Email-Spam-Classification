# Email Spam Classification — Internship Task 1
# Author: Shawaiz

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Step 1: Locate Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'emails.csv')

print("📂 Current directory:", BASE_DIR)
print("📄 Files in directory:", os.listdir(BASE_DIR))
print("📁 Full path to dataset:", DATA_PATH)
print("✅ File exists:", os.path.exists(DATA_PATH))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Step 2: Load Dataset
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully.")

# Step 3: Basic Cleaning
df.dropna(inplace=True)
df.columns = df.columns.str.lower()

if 'label' not in df.columns or 'text' not in df.columns:
    raise ValueError("Dataset must contain 'label' and 'text' columns.")

# Step 4: Encode Labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['text']
y = df['label']

# Step 5: Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 7: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 8: Evaluate Model
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Step 9: Save Model and Vectorizer
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n💾 Model and vectorizer saved.")
