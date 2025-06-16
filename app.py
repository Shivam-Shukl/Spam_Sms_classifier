from flask import Flask, request, render_template
import pickle
import string
import re
import os

app = Flask(__name__)

# ============================
# Load vectorizer and model
# ============================
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please ensure these files are in the same directory as app.py")
    exit() # Exit the application if essential files are missing

# ============================
# Simple stopwords list (can be extended)
# ============================
basic_stopwords = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

# ============================
# Simple stemmer
# ============================
def simple_stem(word):
    # This is a very basic stemming. For more robust stemming without NLTK,
    # you'd typically need a more complex rule-based system or a pre-computed dictionary.
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word

# ============================
# Text preprocessing function
# ============================
def transform_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Find all words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', text)
    # Remove stopwords and apply simple stemming
    processed_words = [simple_stem(word) for word in words if word not in basic_stopwords]
    return " ".join(processed_words)

# ============================
# Flask routes
# ============================
@app.route('/', methods=['GET', 'POST']) # Handles both initial load (GET) and form submission (POST)
def home():
    prediction_result = None # Initialize to None, so it doesn't show initially
    original_message = ""

    if request.method == 'POST':
        input_sms = request.form['userMessage'] # 'userMessage' is the name attribute from your textarea in sms.html
        original_message = input_sms

        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        prediction_result = "Spam" if result == 1 else "Not Spam"

    # Render the sms.html template, passing the variables
    # These variables will be available in sms.html
    return render_template('sms.html',
                           prediction_result=prediction_result,
                           original_message=original_message)

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for easier development
