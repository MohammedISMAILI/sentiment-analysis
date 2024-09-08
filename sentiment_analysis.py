
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the dataset (replace 'reviews_dataset.csv' with the actual file path)
df = pd.read_csv('reviews_dataset.csv')

# Display the first few rows of the dataset
print("Initial dataset sample:")
print(df.head())

# Step 2: Data Cleaning - drop null values and clean the text
df.dropna(subset=['Text', 'Score'], inplace=True)

# Function to clean the text data
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply the text cleaning function
df['cleaned_text'] = df['Text'].apply(clean_text)

# Display the cleaned text sample
print("Cleaned text sample:")
print(df['cleaned_text'].head())

# Step 3: Feature Engineering - Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Define the target variable (1 for positive sentiment, 0 for negative)
y = df['Score'].apply(lambda rating: 1 if rating >= 3 else 0)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predict on the test set using Logistic Regression
y_pred_log = log_model.predict(X_test)

# Step 6: Evaluate Logistic Regression model
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

# Step 7: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Step 8: Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Step 9: Visualize the word frequency from CountVectorizer
word_freq = vectorizer.vocabulary_

# Plot the top 20 words by frequency
plt.figure(figsize=(10, 6))
plt.barh(list(word_freq.keys())[:20], list(word_freq.values())[:20], color='skyblue')
plt.xlabel('Word Frequency')
plt.ylabel('Words')
plt.title('Top 20 Words by Frequency')
plt.show()

# Step 10: Visualize the Random Forest feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(rf_model.feature_importances_)), rf_model.feature_importances_, color='lightcoral')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature Index')
plt.show()
