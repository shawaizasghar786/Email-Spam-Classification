import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
df = pd.read_csv("emails.csv")  # Change to your file name

# Step 2: Split Features and Target
X = df.drop(columns=['Prediction', 'Email No.'], errors='ignore')  # Remove label and any IDs
y = df['Prediction']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = MultinomialNB()  # Best for text count features
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Predict on New Email (Sample)
sample = X.iloc[0].values.reshape(1, -1)  # Use any row as a new input
print("\n🔎 Sample Prediction:", "Spam" if model.predict(sample)[0] == 1 else "Not Spam")
