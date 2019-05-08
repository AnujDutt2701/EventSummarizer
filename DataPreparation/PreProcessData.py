import os
import re
import string

import nltk
import numpy as np
import Article_Parser as scrapper
import pandas
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import preprocessing
import StanfordNER.StanfordNER as stn_ner




englishStemmer=SnowballStemmer("english")

eng_stopwords = set(stopwords.words('english'))
exclude = set(string.punctuation)
lematizer = WordNetLemmatizer()


def clean(doc):
    doc = ''.join(doc.split(',')[1:])
    doc = re.sub('\d+', '', doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in eng_stopwords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # normalized = " ".join(englishStemmer.stem(lematizer.lemmatize(word)) for word in punc_free.split())
    # normalized = " ".join(lematizer.lemmatize(word) for word in punc_free.split())
    return punc_free


def read_data(file, take_rows = 2000, inculde_vac = True):
    raw_data = pandas.read_csv(file)
    raw_data = raw_data.head(take_rows)
    if inculde_vac:
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
    # df['notes'] = df['notes'].apply(lambda x: x.split(',')[1:])
    # df['notes'] = [''.join(l) for l in df['notes']]
    # df['notes'] = df['notes'].str.replace('\d+', '')
    df['notes'] = [clean(doc).split() for doc in df['notes']]
    df['notes'] = [' '.join(l) for l in df['notes']]
    df['notes'].replace('', np.nan, inplace=True)
    df.dropna(subset=['notes'], inplace=True)
    df['label'] = raw_data['event_type']
    return df


def read_data_sub_event(take_rows = 3000, inculde_vac = True):
    sub_events_folder = "../../DataPreparation/SubEventTypes"
    all_data = pandas.DataFrame()
    for root, directory, files in os.walk(sub_events_folder):
        for file in files:
            if '.csv' in file:
                data = pandas.read_csv(os.path.join(root, file))
                data = data.head(take_rows)
                all_data = all_data.append(data, ignore_index=True)
    encoder = preprocessing.LabelEncoder()
    all_data['sub_event_type'] = all_data['sub_event_type'].str.lower()
    all_data['sub_event_type'] = encoder.fit_transform(all_data['sub_event_type'])
    print(encoder.classes_)
    df = pandas.DataFrame()
    df['notes'] = all_data['notes']
    df['notes'] = [clean(doc).split() for doc in df['notes']]
    df['notes'] = [' '.join(l) for l in df['notes']]
    df['notes'].replace('', np.nan, inplace=True)
    df.dropna(subset=['notes'], inplace=True)
    df['label'] = all_data['sub_event_type']
    return df


def article_data():
    elections_data_file = "../DataPreparation/Elections-train.csv"
    elections_data_raw = pandas.read_csv(elections_data_file)
    elections_data = pandas.DataFrame(columns=['X', 'Y'])
    elections_data["X"] = elections_data_raw["notes"]
    elections_data["Y"] = np.ones((len(elections_data_raw),), dtype=int)
    non_elections_data_file = "../DataPreparation/Non-Elections-train.csv"
    non_elections_data = pandas.read_csv(non_elections_data_file)
    all_data = pandas.DataFrame()
    all_data = all_data.append(elections_data, ignore_index=True)
    all_data = all_data.append(non_elections_data, ignore_index=True)

    # for root, directory, files in os.walk(sub_events_folder):
    #     for file in files:
    #         if '.csv' in file:
    #             data = pandas.read_csv(os.path.join(root, file))
    #             data = data.head(take_rows)
    #             all_data = all_data.append(data, ignore_index=True)
    encoder = preprocessing.LabelEncoder()
    # all_data['Y'] = all_data['Y'].str.lower()
    all_data['Y'] = encoder.fit_transform(all_data['Y'])
    print(encoder.classes_)
    df = pandas.DataFrame()
    df['X'] = all_data['X']
    df['X'] = [clean(doc).split() for doc in df['X']]
    df['X'] = [' '.join(l) for l in df['X']]
    df['X'].replace('', np.nan, inplace=True)
    df.dropna(subset=['X'], inplace=True)
    df['Y'] = all_data['Y']
    return df
# print(read_data_sub_event())

#print(clean('On 9 March, students of the Scheduled Caste demonstrated in Jalandhar city (Punjab) to demand the 23 asdfsadf234123 release of post-matric scholarship funds. [size=no report]'))


exampleArray = ['The incredibly intimidating NLP scares people away who are sissies.']

contentArray = ['Starbucks on 10-04-2019 is not doing very well lately.']


##let the fun begin!##
def processLanguage():
    #try:
    for item in contentArray:
        tokenized = nltk.word_tokenize(item)
        tagged = nltk.pos_tag(tokenized)
        print(tagged)

        namedEnt = nltk.ne_chunk(tagged)
        # namedEnt.draw()

        #time.sleep(1)

    # except Exception, e:
    #     print
    #     str(e)


# processLanguage()


# from nltk.parse.corenlp import CoreNLPDependencyParser
# dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
# parses = dep_parser.parse('What is the airspeed of an unladen swallow ?'.split())
# print(parses)