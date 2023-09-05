import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from general_functions import replaceCommonWords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


with open('nb_intents.json') as file:
    data = json.load(file)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text,tag=''):
    text = text.lower()

    if(replaceCommonWords(text)!='' and tag!='greeting' or tag!=''):
        text = replaceCommonWords(text)

    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in string.punctuation and token not in stop_words]
    return ' '.join(tokens)

# Process data
corpus = []
intents = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        preprocessed_text = preprocess_text(pattern,intent['tag'])
        corpus.append(preprocessed_text)
        intents.append(intent['tag'])

# Text vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, intents, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'fit_prior': [True, False],
    'class_prior': [None, [0.5, 0.5], [0.3, 0.7]]  # Example class priors
}
model = MultinomialNB()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# Get the best parameters
best_params = grid_search.best_params_

# Prediction using best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

def nbAnswer(user_input=''):
        
    predicted_intent        = None
    preprocessed_question   = preprocess_text(user_input)
    question_vector         = vectorizer.transform([preprocessed_question])
    predicted_intent        = best_model.predict(question_vector)[0]

    if predicted_intent:
        response = predicted_intent        
    else:
        response = "I'm sorry, I don't have the answer to that question."

    return response, accuracy



# while 0!=1:
#     user_input = input("Enter a text: ")
#     print(nbAnswer(user_input))






