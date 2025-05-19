import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("Case_Information.csv", encoding="ISO-8859-1")

# Combine relevant text fields
df_text = df[['region', 'province', 'muni_city', 'health_status']].dropna()
df_text['text_column'] = df_text[['region', 'province', 'muni_city']].agg(' '.join, axis=1)

# Clean text function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)      # Replace multiple spaces with one
    return text.lower().strip()           # Lowercase and trim

# Apply cleaning
df_text['cleaned_text'] = df_text['text_column'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df_text['label'] = label_encoder.fit_transform(df_text['health_status'])

# Feature and target
X = df_text['cleaned_text']
y = df_text['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Model evaluation function
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
    return pd.DataFrame(results)

# Evaluate and identify best model
model_comparisons = evaluate_models(models, X_train_tfidf, X_test_tfidf, y_train, y_test)
best_model = model_comparisons.loc[model_comparisons['F1-Score'].idxmax()]

# Print results
print("Model Comparison Results:")
print(model_comparisons)
print(f"\nChosen Model: {best_model['Model']} (F1-Score: {best_model['F1-Score']:.2f})")
