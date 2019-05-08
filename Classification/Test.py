import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from scipy.sparse import coo_matrix, hstack
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords


data_file = "../DataPreparation/2019-01-01-2019-03-25-Southern_Asia.csv" #"../DataPreparation/2017-03-17-2018-03-17-VAC.csv"
classifier_pkl_file = "classifier.p"
tokenizer_pkl_file = "tokenizer.p"


def load_from_pkl_file(file):
    return pickle.load(open(file, "rb"))


eng_stopwords = set(stopwords.words('english'))
exclude = set(string.punctuation)
lematizer = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in eng_stopwords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lematizer.lemmatize(word) for word in punc_free.split())
    return normalized


def test_with_new_data(file):
    test_data = pd.read_csv(file)
    tfidf_ngrams_vect = load_from_pkl_file(tokenizer_pkl_file)
    print(tfidf_ngrams_vect.get_feature_names())
    test_data['notes'] = test_data['notes'].apply(lambda x: x.split(',')[1:])
    test_data['notes'] = [''.join(l) for l in test_data['notes']]
    test_data['notes'] = test_data['notes'].str.replace('\d+', '')
    test_data['notes'] = [clean(doc).split() for doc in test_data['notes']]
    test_data['notes'] = [' '.join(l) for l in test_data['notes']]
    tfidf_input_vector = tfidf_ngrams_vect.transform(test_data['notes'])
    print(test_data['notes'][0])
    print(tfidf_input_vector)
    test_shape = tfidf_input_vector.shape
    print(test_shape)
    # if test_shape[1] < 10000:
    #     zeros = np.zeros([test_shape[0], 10000-test_shape[1]], dtype = int)
    #     zeros_df = coo_matrix(zeros)
    #     print(tfidf_input_vector.shape)
    #     tfidf_input_vector = hstack([tfidf_input_vector, zeros_df]).toarray()
    #     print(tfidf_input_vector.shape)
    return tfidf_input_vector


def predict_output(classifier, vector):
    return classifier.predict(vector)


input_vector = test_with_new_data(data_file)
model = load_from_pkl_file(classifier_pkl_file)
output_vector = predict_output(model, input_vector)
print(output_vector)
input_count = input_vector.shape[0]

