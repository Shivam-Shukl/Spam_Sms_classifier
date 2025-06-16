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
    # It's good practice to use absolute paths in production, especially if your app
    # might be run from a different working directory.
    # __file__ gives the path to the current script.
    # os.path.dirname(__file__) gives the directory containing the script.
    # os.path.join combines path components intelligently.
    script_dir = os.path.dirname(__file__)
    vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
    model_path = os.path.join(script_dir, 'model.pkl')

    tfidf = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please ensure these files are in the same directory as app.py")
    # In a production environment, you might want to log this error
    # and potentially gracefully shut down or return a server error page.
    exit(1) # Use exit(1) to indicate an error state

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
    # For production deployment, you would NOT use app.run() directly like this.
    # Instead, you'd use a WSGI server (e.g., Gunicorn, Waitress) to run the app.
    # Example for Gunicorn: gunicorn --bind 0.0.0.0:5000 app:app
    # The 'app:app' means 'app.py' module and 'app' Flask instance.
    
    # This block is primarily for local testing without a WSGI server.
    # For actual production, this `if __name__ == '__main__':` block might be
    # entirely removed or only contain a placeholder for local execution.
    # We remove debug=True and explicitly bind to 0.0.0.0 to be ready for
    # how a WSGI server often expects the app to be set up.
    app.run(host='0.0.0.0', port=5000)
