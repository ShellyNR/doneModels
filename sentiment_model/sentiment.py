import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Constants
PERCENTAGES = 100
HIGH_DESCRIPTION = "The description sentiment is great!"
MEDIUM_DESCRIPTION = "The description sentiment is positive, but you can consider improving it a little."
LOW_DESCRIPTION = "The description sentiment is negative, try using positive words to describe the property."
HIGH_GRADE = 85
MEDIUM_GRADE = 60


# infer function on given description
def infer(description, training_vector, logistic_regression):
    text = [description]
    text_dtm = training_vector.transform(text)
    prediction = logistic_regression.predict_proba(text_dtm)

    prediction = prediction[0]
    # prediction = [Negative,Positive]
    return prediction[1]


# the function gets a grade (0 - 1) and returns a sentence regarding the grade
def sentence_grade(grade):
    grade = grade * PERCENTAGES
    if grade > HIGH_GRADE:
        return HIGH_DESCRIPTION
    if grade > MEDIUM_GRADE:
        return MEDIUM_DESCRIPTION
    else:
        return LOW_DESCRIPTION


def sentiments_model(description):
    training_vector_filename = 'sentiment_model/training_vector.sav'
    training_vector = pickle.load(open(training_vector_filename, 'rb'))

    logistic_regression_filename = 'sentiment_model/logistic_regression.sav'
    logistic_regression = pickle.load(open(logistic_regression_filename, 'rb'))

    grade = infer(description, training_vector, logistic_regression)
    return [grade, sentence_grade(grade)]
