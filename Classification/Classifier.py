import pandas
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import Sequential
from keras.layers import Activation
import numpy as np
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from sklearn import svm

# print(eng_stopwords)

# TODO: This is to be altered
eng_stopwords = set(stopwords.words('english'))
exclude = set(string.punctuation)
lematizer = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in eng_stopwords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lematizer.lemmatize(word) for word in punc_free.split())
    return normalized


def read_data(file):
    raw_data = pandas.read_csv(file)
    raw_data = raw_data.head(2000)
    raw_data2 = pandas.read_csv('../DataPreparation/2016-03-18-2019-03-18-OnlyVAC.csv')
    raw_data = raw_data.append(raw_data2, ignore_index=True)
    raw_data = raw_data.sample(frac=1)
    raw_data.loc[raw_data['event_type'].isin(['Protests','Riots']), 'event_type'] = 'Riots/Protests'
    encoder = preprocessing.LabelEncoder()
    raw_data['event_type'] = encoder.fit_transform(raw_data['event_type'])
    print("raw event type observed by encoder")
    print(encoder.classes_)
    df = pandas.DataFrame()
    df['notes'] = raw_data['notes']
    # Removes the data in the start of the notes
    df['notes'] = df['notes'].apply(lambda x: x.split(',')[1:])
    df['notes'] = [''.join(l) for l in df['notes']]
    df['notes'] = df['notes'].str.replace('\d+', '')
    df['notes'] = [clean(doc).split() for doc in df['notes']]
    df['notes'] = [' '.join(l) for l in df['notes']]

    df['label'] = raw_data['event_type']
    df['label'] = np_utils.to_categorical(np.array(df['label']), 2, dtype='int')
    print(df)
    return df['notes'], df['label']


def process_data(X,Y):
    return model_selection.train_test_split(X, Y, test_size=0.2)


def generate_model(no_of_input_nodes):
    model = Sequential()
    model.add(Dense(512, input_dim=no_of_input_nodes))
    model.add(Activation('relu'))
    model.add(Dense(512, input_dim=512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(optimizer='adamax',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_model_svm(kernal, regularizer, gamma, training_data, training_target):
    svm_classifier = svm.SVC(kernel=kernal , C=regularizer, gamma=gamma)
    svm_classifier.fit(training_data, training_target)
    return svm_classifier


all_X, all_Y = read_data('../DataPreparation/2018-03-17-2019-03-17.csv')
train_X, test_X, train_Y, test_Y = process_data(all_X, all_Y)

tfidf_ngrams_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                     max_df=0.5,
                                    stop_words=eng_stopwords)
tfidf_ngrams_vect.fit_transform(all_X)
print("The feature words are: ")
print((tfidf_ngrams_vect.get_feature_names()))
pickle.dump(tfidf_ngrams_vect, open("tokenizer.p","wb"))
train_X_vector = tfidf_ngrams_vect.transform(train_X)
test_X_vector = tfidf_ngrams_vect.transform(test_X)


print("Training started")
# SVM Model
classifier_model = generate_model_svm("rbf", 1, 0.01, train_X_vector, train_Y)
predicted_value = classifier_model.predict(train_X_vector)

# Naive Bayes
# classifier_model = generate_model(train_X_vector.shape[1])
# classifier_model = classifier_model.fit(train_X_vector, train_Y)
# predicted_value = classifier_model.model.predict(test_X_vector)

pickle.dump(classifier_model, open("classifier.p","wb"))

right = 0
wrong = 0

print("for unseen data")
for i, j in zip(predicted_value, train_Y):
    print(" predicted value is " + str(i) + " actual value is " + str(j))
    if i == j:
        right = right + 1
    else:
        wrong = wrong + 1
print("Accuracy is: " + str(right/(right + wrong) * 100))
