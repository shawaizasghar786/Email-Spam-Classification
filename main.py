# Email Spam Classification — Internship Task 1
# Author: Shawaiz
# Description: Classifies emails as spam or not spam using a labeled dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Step 1: Load Dataset (relative path)
DATA_PATH = 'emails.xlsx'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_excel(DATA_PATH)
print("✅ Dataset loaded successfully.")

# Step 2: Basic Cleaning
df.dropna(inplace=True)
df.columns = df.columns.str.lower()

# Ensure required columns exist
if 'label' not in df.columns or 'text' not in df.columns:
    raise ValueError("Dataset must contain 'label' and 'text' columns.")

# Step 3: Encode Labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['text']
y = df['label']

# Step 4: Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Step 8: Save Model and Vectorizer
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n💾 Model and vectorizer saved.")
