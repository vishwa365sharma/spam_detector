
# =====================================
# SPAM MAIL DETECTOR PROJECT
# =====================================

# Import Libraries

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from google.colab import files


Uploaded = files.upload()
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]

df.columns = ['Label', 'Message']

print(df.head())


# Data Information

print(df.info())

print(df['Label'].value_counts())


# Features and Target

X = df["Message"]

y = df["Label"]


# Convert Text into Numbers

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Naive Bayes


nb = MultinomialNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

acc_nb = accuracy_score(y_test, pred_nb)

print("\n Naive Bayes ")

print("Accuracy :", acc_nb)

print(confusion_matrix(y_test, pred_nb))

print(classification_report(y_test, pred_nb))


# Logistic Regression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)

print("\n Logistic Regression ")

print("Accuracy :", acc_lr)

print(confusion_matrix(y_test, pred_lr))

print(classification_report(y_test, pred_lr))



# Accuracy Comparison


print("\n Accuracy Comparison ")

print("Naive Bayes :", round(acc_nb*100,2), "%")

print("Logistic Regression:", round(acc_lr*100,2), "%")


#  Best Model


if acc_nb > acc_lr:
    model = nb
    print("Best Model : Naive Bayes")
else:
    model = lr
    print("Best Model : Logistic Regression")



# Save Model


pickle.dump(model, open("spam_model.pkl","wb"))

pickle.dump(vectorizer, open("tfidf.pkl","wb"))

print("Model Saved Successfully")

sample = ["congratulations brother"]
sample = vectorizer.transform(sample)
prediction = model.predict(sample)
print(prediction)
