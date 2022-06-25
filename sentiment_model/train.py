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
PATH = 'sentiment_data/opinions.tsv'
LANGUAGE = 'english'
PERCENTAGES = 100
HIGH_DESCRIPTION = "The description sentiment is great!"
MEDIUM_DESCRIPTION = "The description sentiment is positive, but you can consider improving it a little."
LOW_DESCRIPTION = "The description sentiment is negative, try using positive words to describe the property."
HIGH_GRADE = 85
MEDIUM_GRADE = 60
K_KNN = 3
TEST_SIZE = 0.2


def train_all_models():
    values = load_data()
    naive_bayes_model(values)
    logistic_regression_model(values)
    svm_model(values)
    knn_model(values)
    return values['x'], values['y']

# loads the data from the dataset and defines the train and validation
def load_data():
    values = dict()
    data = pd.read_table(PATH, header=None, skiprows=1, names=['Sentiment', 'Review'])
    values['x'] = data.Review
    values['y'] = data.Sentiment

    # Using CountVectorizer to convert text into tokens/features
    values['vector'] = CountVectorizer(stop_words=LANGUAGE, ngram_range=(1, 1), max_df=.80, min_df=4)
    x_train, x_test, y_train, y_test = train_test_split(values['x'], values['y'], random_state=1, test_size=TEST_SIZE)
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

# Train the best performing model (Logistic Regression)
def logistic_regression_train():
    values = load_data()
    training_vector = CountVectorizer(stop_words=LANGUAGE, ngram_range=(1, 1), max_df=.80, min_df=5)
    training_vector.fit(values['x'])
    x_dtm = training_vector.transform(values['x'])
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_dtm, values['y'])
    training_vector_filename = 'training_vector.sav'
    pickle.dump(training_vector, open(training_vector_filename, 'wb'))

    logistic_regression_filename = 'logistic_regression.sav'
    pickle.dump(logistic_regression, open(logistic_regression_filename, 'wb'))

