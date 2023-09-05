import json
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve  # Import learning_curve
import pandas as pd
import seaborn as sns
from general_functions import replaceCommonWords  # Replace this with your actual import for general_functions

# Load the JSON dataset
with open('intents.json', 'r') as file:
    data = json.load(file)

# Extract patterns, corresponding intents, and responses
patterns = []
intents = []
responses = {}
for intent_data in data['intents']:
    patterns.extend(intent_data['patterns'])
    intents.extend([intent_data['tag']] * len(intent_data['patterns']))
    responses[intent_data['tag']] = intent_data['responses']

# Split the data into training, validation, and testing sets (80-10-10)
X_train, X_temp, y_train, y_temp = train_test_split(patterns, intents, test_size=0.2, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert text data to numerical feature vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_validation_tfidf = tfidf_vectorizer.transform(X_validation)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Hyperparameter tuning for SVM
best_accuracy = 0
best_svm = None
for C in [0.1, 1, 10]:
    for kernel in ['linear', 'rbf', 'poly']:
        svm_classifier = SVC(C=C, kernel=kernel)
        svm_classifier.fit(X_train_tfidf, y_train)
        y_pred = svm_classifier.predict(X_validation_tfidf)
        accuracy = accuracy_score(y_validation, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_svm = svm_classifier

# Print the best SVM model's accuracy on the validation set
print("Accuracy on Validation Set:", best_accuracy)

# Evaluate the best SVM model on the test set
y_pred = best_svm.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", test_accuracy)

# Print the accuracy of the model as a percentage
print(f"Accuracy on Test Set: {test_accuracy * 100:.2f}%")

# Generate classification report for the test set
report = classification_report(y_test, y_pred, output_dict=True)

# Create a heatmap for the classification report
fig, ax = plt.subplots(figsize=(8, 6))
report_df = pd.DataFrame(report).transpose()
sns.heatmap(report_df, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
plt.title("Classification Report")

# Save the classification report as an image
plt.savefig('classification_report.png')

# Show the classification report
plt.show()

# Generate confusion matrix as an image with values
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_svm, X_train_tfidf, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Create learning curve plots
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# Response time analysis for 100 randomly selected questions
response_times = []

for _ in range(100):
    user_input = random.choice(patterns)  # Randomly select a question
    start_time = time.time()

    # Convert user input to TF-IDF vector and predict intent
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    predicted_intent = best_svm.predict(user_input_tfidf)

    end_time = time.time()
    response_time = end_time - start_time
    response_times.append(response_time)

    # Print the predicted intent and response
    print("You:", user_input)
    print("Predicted Intent:", predicted_intent[0])
    if predicted_intent[0] in responses:
        response_options = responses[predicted_intent[0]]
        if response_options:
            print("Chatbot:", response_options[0])
        else:
            print("Chatbot: I'm not sure how to respond to that.")
    else:
        print("Chatbot: I'm not sure how to respond to that intent.")

# Plot response time histogram
plt.hist(response_times, bins=20)
plt.xlabel("Response Time (seconds)")
plt.ylabel("Frequency")
plt.title("Response Time Analysis")
plt.show()
