import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_all_models():
    X_train_dtm, y_train, X_test_dtm, y_test, X_test, vect, X, y = load_data()
    naive_bayes_model(X_train_dtm, y_train, X_test_dtm, y_test)
    logistic_regression_model(X_train_dtm, y_train, X_test_dtm, y_test)
    svm_model(X_train_dtm, y_train, X_test_dtm, y_test)
    knn_model(X_train_dtm, y_train, X_test_dtm, y_test)
    return X, y

def load_data():
    path = './sentiment_data/opinions.tsv'
    data = pd.read_table(path, header=None, skiprows=1, names=['Sentiment', 'Review'])
    X = data.Review
    y = data.Sentiment

    # Using CountVectorizer to convert text into tokens/features
    vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Using training data to transform text into counts of features for each message
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    return X_train_dtm, y_train, X_test_dtm, y_test, X_test, vect, X, y

# Accuracy using Naive Bayes Model
def naive_bayes_model(X_train_dtm, y_train, X_test_dtm, y_test):
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    y_pred = NB.predict(X_test_dtm)
    accuracy = metrics.accuracy_score(y_test, y_pred)*100
    print('\nNaive Bayes')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy, NB, y_pred

# Accuracy using Logistic Regression Model
def logistic_regression_model(X_train_dtm, y_train, X_test_dtm, y_test):
    LR = LogisticRegression()
    LR.fit(X_train_dtm, y_train)
    y_pred = LR.predict(X_test_dtm)
    accuracy = metrics.accuracy_score(y_test, y_pred)*100
    print('\nLogistic Regression')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Accuracy using SVM Model
def svm_model(X_train_dtm, y_train, X_test_dtm, y_test):
    SVM = LinearSVC()
    SVM.fit(X_train_dtm, y_train)
    y_pred = SVM.predict(X_test_dtm)
    accuracy = metrics.accuracy_score(y_test, y_pred)*100
    print('\nSupport Vector Machine')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Accuracy using KNN Model
def knn_model(X_train_dtm, y_train, X_test_dtm, y_test):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train_dtm, y_train)
    y_pred = KNN.predict(X_test_dtm)
    accuracy = metrics.accuracy_score(y_test, y_pred)*100
    print('\nK Nearest Neighbors (NN = 3)')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Custom Test: Test a review on the best performing model (Logistic Regression)
def logistic_regression_train(X, y):
    trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
    trainingVector.fit(X)
    X_dtm = trainingVector.transform(X)
    LR_complete = LogisticRegression()
    LR_complete.fit(X_dtm, y)

    return trainingVector, LR_complete

# infer function on given description
def infer(description, trainingVector, LR_complete):
    text = [description]
    text_dtm = trainingVector.transform(text)
    prediction = LR_complete.predict_proba(text_dtm)

    # tags = ['Negative','Positive']
    # Display Output
    print('The review is predicted', prediction[0])

    prediction = prediction[0]
    return prediction[1]

# the function gets a grade (0 - 1) and returns a sentence regarding the grade
def sentence_grade(grade):
    grade = grade * 100
    if grade > 85:
        return "The description sentiment is great!"
    if grade > 60:
        return "The description sentiment is positive, but you can consider improving it a little."
    else:
        return "The description sentiment is negative, try using positive words to describe the property."

def sentiments_model(description):
    # X, y = train_all_models()
    X_train_dtm, y_train, X_test_dtm, y_test, X_test, vect, X, y = load_data()
    trainingVector, LR_complete = logistic_regression_train(X, y)
    grade = infer(description, trainingVector, LR_complete)
    return [grade, sentence_grade(grade)]

