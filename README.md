# Spam Classifier – InboxGuard

InboxGuard is a simple yet powerful spam classification web application. It leverages machine learning to detect whether a given SMS message is **Spam** or **Not Spam**. Built with Flask and scikit-learn, it provides an intuitive web interface where users can input any message and receive a prediction instantly.

**Live Demo**: [https://inboxguard.onrender.com/](https://inboxguard.onrender.com/)

---

## Features

- Detects spam messages using Natural Language Processing (NLP)
- Trained using Multinomial Naive Bayes and TF-IDF Vectorizer
- Clean and minimal web interface built with Flask
- Deployed on Render for easy public access
- Lightweight, modular, and easy to extend

---

## Screenshots

| Input Page | Prediction Page |
|------------|-----------------|
| ![Input](https://github.com/user-attachments/assets/115f6bb0-ba16-4fac-b3b5-4fc25ad42e7a) | ![Prediction](https://github.com/user-attachments/assets/a0686ba9-6cd2-4cbe-8bf6-4761afd3d824) |

---

## How It Works

1. **Data Preprocessing**: The SMS text data is cleaned and converted into numerical features using `TfidfVectorizer`.
2. **Model Training**: A `Multinomial Naive Bayes` classifier is trained on the preprocessed data.
3. **Prediction Pipeline**: The model and vectorizer are serialized with `pickle` and used in a Flask web app to make real-time predictions.
4. **Deployment**: The app is deployed using [Render](https://render.com) and accessible via a public URL.

---


## Project Structure

This project is developed in the `master` branch. The files include:

spam-classifier/
│
├── app.py # Main Flask application
├── spam_spam_classifier.ipynb # Jupyter Notebook for preprocessing and model training
├── model.pkl # Trained spam classification model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── templates/
│ └── index.html # Frontend HTML template
├── static/
│ └── style.css # Optional CSS styles
├── Screenshot 2025-05-14 140607.png # Screenshot: Input page
├── Screenshot 2025-05-14 140645.png # Screenshot: Output page
├── requirements.txt # List of dependencies
├── .gitignore # Git ignore file
├── nltk_data # NLTK data for text processing
└── README.md # Project documentation
