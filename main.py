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

# Step 3: Clean and Inspect Columns
df.columns = df.columns.str.lower()
print("📋 Columns in dataset:", df.columns.tolist())

# Step 4: Drop NaNs and Convert to Strings
df.dropna(subset=['message', 'category'], inplace=True)
df['message'] = df['message'].astype(str)

# Step 5: Encode Labels
df['label'] = df['category'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Step 6: Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 8: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 9: Evaluate Model
y_pred = model.predict(X_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Step 10: Save Model and Vectorizer
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n💾 Model and vectorizer saved.")
