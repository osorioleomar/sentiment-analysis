import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

print("Loading data...")
# Load the data
data = pd.read_csv("dataset.csv")
print("Data loaded.")

# Extract the feature and target variables
X = data["review"]
y = data["sentiment"]

print("Converting text data to numerical data...")
# Convert the text data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
print("Text data converted.")

print("Splitting data into training and testing sets...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print("Data split.")

print("Training the model...")
# Train the model using Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
print("Model trained.")

# Evaluate the model on the test data
accuracy = clf.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save the trained classifier to disk
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save the vectorizer to disk
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved.")
