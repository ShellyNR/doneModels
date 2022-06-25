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
DATASET_PATH = 'sentiment_model/sentiment_data/opinions.tsv'
TRAINING_VECTOR_FILENAME = 'training_vector.sav'
LOGISTIC_REGRESSION_FILENAME = 'logistic_regression.sav'
LANGUAGE = 'english'
PERCENTAGES = 100
K_KNN = 3
TEST_SIZE = 0.2
MAX_DF = .80
MIN_DF = 5

# train all optional models and prints their accuracies
def train_all_models():
    values = load_data()
    split_data(values)
    train_model(values, MultinomialNB(), 'Naive Bayes')
    train_model(values, LogisticRegression(), 'Logistic Regression')
    train_model(values, LinearSVC(), 'Support Vector Machine (SVM)')
    train_model(values, KNeighborsClassifier(n_neighbors=K_KNN), 'K Nearest Neighbors')


# Splits the data to train and validation
def split_data(values):
    values['x_train'], values['x_test'], values['y_train'], values['y_test'] = \
        train_test_split(values['x'], values['y'], random_state=1, test_size=TEST_SIZE)

    # Using CountVectorizer to convert the text into tokens/features
    vector = CountVectorizer(stop_words=LANGUAGE, max_df=MAX_DF, min_df=MIN_DF)
    vector.fit(values['x_train'])
    values['x_train_dtm'] = vector.transform(values['x_train'])
    values['x_test_dtm'] = vector.transform(values['x_test'])


# Loads the data from the dataset and define x and y
def load_data():
    values = dict()
    data = pd.read_table(DATASET_PATH, names=['Sentiment', 'Review'])
    values['x'] = data.Review
    values['y'] = data.Sentiment

    return values


# Accuracy using a given model
def train_model(values, model, model_name):
    model.fit(values['x_train_dtm'], values['y_train'])
    y_prediction = model.predict(values['x_test_dtm'])
    accuracy = metrics.accuracy_score(values['y_test'], y_prediction) * PERCENTAGES
    print('\n' + model_name)
    print('Accuracy Score: ', accuracy, '%', sep='')


# Train the best performing model (Logistic Regression)
def logistic_regression_train():
    values = load_data()
    training_vector = CountVectorizer(stop_words=LANGUAGE, max_df=MAX_DF, min_df=MIN_DF)
    training_vector.fit(values['x'])
    x_dtm = training_vector.transform(values['x'])

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_dtm, values['y'])

    pickle.dump(training_vector, open(TRAINING_VECTOR_FILENAME, 'wb'))
    pickle.dump(logistic_regression, open(LOGISTIC_REGRESSION_FILENAME, 'wb'))
