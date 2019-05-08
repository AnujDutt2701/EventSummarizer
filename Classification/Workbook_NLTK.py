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
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")


eng_stopwords = set(stopwords.words('english'))
exclude = set(string.punctuation)
lematizer = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in eng_stopwords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(englishStemmer.stem(lematizer.lemmatize(word)) for word in punc_free.split())
    return normalized

def read_data(file):
    raw_data = pandas.read_csv(file)
    raw_data = raw_data.head(8000)
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
    print(df)
    return df['notes'], df['label']


notes, label = read_data('../DataPreparation/2018-03-17-2019-03-17.csv')
tfidf_ngrams_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                     max_df=0.5,
                                    stop_words=eng_stopwords)
tfidf_ngrams_vect.fit_transform(notes)
print("The feature words are: ")
feature_words = tfidf_ngrams_vect.get_feature_names()
feature_words_string = ' '.join(feature_words)
tokens = nltk.word_tokenize(feature_words_string)
print(tokens)
tagged = nltk.pos_tag(tokens)
target_tags = ['JJS','RB','VB','VBD','VBG','VBN','VBP','VBZ']
target_tokens = [t for t in tagged if target_tags.__contains__(t[1])]
print(target_tokens)