import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
print("📌 Loading dataset...")
data = pd.read_csv("spam.csv")

# 2️⃣ Remove Non-Numeric Columns (Fixes the Error)
data = data.select_dtypes(include=['number'])

# 3️⃣ Split into Features & Labels
X = data.drop(columns=['Prediction'], errors='ignore')  # Features
y = data['Prediction']  # Target

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6️⃣ Save Model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as spam_model.pkl")

# 7️⃣ Evaluate Model
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8️⃣ Pie Chart of Results
spam_count = sum(y_pred)
not_spam_count = len(y_pred) - spam_count
plt.pie([spam_count, not_spam_count], labels=['Spam', 'Not Spam'], autopct='%1.1f%%', startangle=90)
plt.title("Spam vs Not Spam Prediction Distribution")
plt.show()

# 9️⃣ Manual Email Prediction
print("\n✍️ Type sample input for prediction (comma separated word counts):")
print(f"Example input length should be {X.shape[1]} numbers!")
try:
    user_input = input("Enter values: ")
    sample_input = np.array([list(map(int, user_input.split(',')))])
    prediction = model.predict(sample_input)[0]
    print("🔎 Prediction:", "Spam" if prediction == 1 else "Not Spam")
except:
    print("⚠️ Invalid input format. Please enter comma-separated numbers only.")
