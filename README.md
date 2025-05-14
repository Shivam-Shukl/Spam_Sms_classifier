# Spam Classifier – InboxGuard

InboxGuard is a simple yet powerful spam classification web application. It leverages machine learning to detect whether a given SMS message is **Spam** or **Not Spam**. Built with Flask and scikit-learn, it provides an intuitive web interface where users can input any message and receive a prediction instantly.

 **Live Demo**: [https://inboxguard.onrender.com/](https://inboxguard.onrender.com/)

---

##  Features

- 🔍 Detects spam messages using natural language processing (NLP)
- 🧠 Trained using Multinomial Naive Bayes and TfidfVectorizer
- 🌐 Simple and clean web interface powered by Flask
- ☁️ Deployed on Render for public access
- 📦 Easily reproducible and modifiable

---

## 🖼 Screenshots

| Input Message Page |
|--------------------|-------------------|
| ![Input](https://github.com/user-attachments/assets/115f6bb0-ba16-4fac-b3b5-4fc25ad42e7a)
Prediction Output |
|--------------------|-------------------|
| ![Prediction](https://github.com/user-attachments/assets/a0686ba9-6cd2-4cbe-8bf6-4761afd3d824) |


---

## 🏗 How It Works

1. **Preprocessing**: SMS data is cleaned and transformed using `TfidfVectorizer`.
2. **Model Training**: A `Multinomial Naive Bayes` classifier is trained on the preprocessed data.
3. **Prediction**: The model predicts whether the input message is spam or not.
4. **Deployment**: The model and vectorizer are saved with `pickle` and used in a Flask app.

---


