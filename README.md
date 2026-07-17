Spam Mail Detector


live demo
click here:
https://spamdetector365.streamlit.app/

Overview

AI Spam Mail Detector is a Machine Learning web application that classifies email or SMS messages as Spam or Not Spam (Normal). The application is built using Python, Streamlit, Scikit-learn, and TF-IDF Vectorizer.

Features

- Detects Spam and Ham messages
- User-friendly Streamlit interface
- Fast and accurate predictions
- TF-IDF text vectorization
- Machine Learning classification model
- Responsive and clean UI

Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Pickle

Project Structure

Spam-Mail-Detector/
│── app.py
│── spam_model.pkl
│── tfidf.pkl
│── spam.csv
│── requirements.txt
└── README.md

Installation

1. Clone the repository:

git clone <repository-link>

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run app.py

How It Works

1. Enter an email or SMS message.
2. Click the Predict button.
3. The model classifies the message as:
   - 🚨 Spam
   - ✅ Not Spam (Normal)

Machine Learning Model

- TF-IDF Vectorizer
- Logistic Regression / Multinomial Naive Bayes (Best Performing Model)

Future Improvements

- Spam probability score
- Email file upload support
- Multiple language support
- Dark/Light theme
- Message history

Author

Vishwa Sharma

---

Made with ❤️ using Python and Streamlit.
