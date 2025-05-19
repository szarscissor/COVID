import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Case_Information.csv", encoding="ISO-8859-1")

# Combine location fields into text
df_text = df[['region', 'province', 'muni_city', 'health_status']].dropna()
df_text['text_column'] = df_text[['region', 'province', 'muni_city']].agg(' '.join, axis=1)

# Text cleaning
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

df_text['cleaned_text'] = df_text['text_column'].apply(clean_text)

# Encode target
label_encoder = LabelEncoder()
df_text['label'] = label_encoder.fit_transform(df_text['health_status'])

# Split data
X = df_text['cleaned_text']
y = df_text['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model (Random Forest)
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
