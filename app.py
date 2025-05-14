from flask import Flask, request, render_template
import pickle
import string
import re
import os

app = Flask(__name__)

# ============================
# Load vectorizer and model
# ============================
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

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
# Simple stemmer (optional, very light)
# ============================
def simple_stem(word):
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
    words = re.findall(r'\b\w+\b', text)  # Only keep words
    words = [word for word in words if word not in basic_stopwords]
    words = [simple_stem(word) for word in words]
    return " ".join(words)

# ============================
# Flask routes
# ============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['message']
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    output = "Spam" if result == 1 else "Not Spam"
    return render_template('index.html', prediction_text=f"Prediction: {output}")

if __name__ == '__main__':
    app.run(debug=True)
