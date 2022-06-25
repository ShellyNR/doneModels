import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Constants
PATH = 'sentiment_model/sentiment_data/opinions.tsv'
LANGUAGE = 'english'
PERCENTAGES = 100
HIGH_DESCRIPTION = "The description sentiment is great!"
MEDIUM_DESCRIPTION = "The description sentiment is positive, but you can consider improving it a little."
LOW_DESCRIPTION = "The description sentiment is negative, try using positive words to describe the property."
HIGH_GRADE = 85
MEDIUM_GRADE = 60
K_KNN = 3

def train_all_models():
    values = load_data()
    naive_bayes_model(values)
    logistic_regression_model(values)
    svm_model(values)
    knn_model(values)
    return values['x'], values['y']

def load_data():
    values = dict()
    data = pd.read_table(PATH, header=None, skiprows=1, names=['Sentiment', 'Review'])
    values['x'] = data.Review
    values['y'] = data.Sentiment

    # Using CountVectorizer to convert text into tokens/features
    values['vector'] = CountVectorizer(stop_words=LANGUAGE, ngram_range=(1, 1), max_df=.80, min_df=4)
    x_train, x_test, y_train, y_test = train_test_split(values['x'], values['y'], random_state=1, test_size=0.2)
    values['x_train'] = x_train
    values['x_test'] = x_test
    values['y_train'] = y_train
    values['y_test'] = y_test
    # Using training data to transform text into counts of features for each message
    values['vector'].fit(x_train)
    values['x_train_dtm'] = values['vector'].transform(x_train)
    values['x_test_dtm'] = values['vector'].transform(x_test)

    return values

# Accuracy using Naive Bayes Model
def naive_bayes_model(values):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(values['x_train_dtm'], values['y_train'])
    y_prediction = naive_bayes.predict(values['x_test_dtm'])
    accuracy = metrics.accuracy_score(values['y_test'], y_prediction) * PERCENTAGES
    print('\nNaive Bayes')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy, naive_bayes, y_prediction

# Accuracy using Logistic Regression Model
def logistic_regression_model(values):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(values['x_train_dtm'], values['y_train'])
    y_prediction = logistic_regression.predict(values['x_test_dtm'])
    accuracy = metrics.accuracy_score(values['y_test'], y_prediction) * PERCENTAGES
    print('\nLogistic Regression')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Accuracy using SVM Model
def svm_model(values):
    SVM = LinearSVC()
    SVM.fit(values['x_train_dtm'], values['y_train'])
    y_prediction = SVM.predict(values['x_test_dtm'])
    accuracy = metrics.accuracy_score(values['y_test'], y_prediction) * PERCENTAGES
    print('\nSupport Vector Machine')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Accuracy using KNN Model
def knn_model(values):
    KNN = KNeighborsClassifier(n_neighbors=K_KNN)
    KNN.fit(values['x_train_dtm'], values['y_train'])
    y_prediction = KNN.predict(values['x_test_dtm'])
    accuracy = metrics.accuracy_score(values['y_test'], y_prediction) * PERCENTAGES
    print('\nK Nearest Neighbors (NN = ' + str(K_KNN) + ')')
    print('Accuracy Score: ', accuracy, '%', sep='')
    return accuracy

# Custom Test: Test a review on the best performing model (Logistic Regression)
def logistic_regression_train(x, y):
    training_vector = CountVectorizer(stop_words=LANGUAGE, ngram_range=(1, 1), max_df=.80, min_df=5)
    training_vector.fit(x)
    x_dtm = training_vector.transform(x)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_dtm, y)

    return training_vector, logistic_regression

# infer function on given description
def infer(description, training_vector, logistic_regression):
    text = [description]
    text_dtm = training_vector.transform(text)
    prediction = logistic_regression.predict_proba(text_dtm)

    # tags = ['Negative','Positive']
    # Display Output
    # print('The review is predicted', prediction[0])

    prediction = prediction[0]
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
    # x, y = train_all_models()
    values = load_data()
    training_vector, logistic_regression = logistic_regression_train(values['x'], values['y'])
    grade = infer(description, training_vector, logistic_regression)
    return [grade, sentence_grade(grade)]
