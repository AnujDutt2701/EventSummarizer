import gensim
from gensim import corpora
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn import svm
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
import seaborn as sb
import matplotlib.pyplot as plt


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def cleandoc():
    df = pd.read_csv('../DataPreparation/2018-03-17-2019-03-17.csv')
    # Reformat values for column a using an unnamed lambda function
    df['notes'] = df['notes'].apply(lambda x: x.split(',')[1:])
    df['notes'] = [''.join(l) for l in df['notes']]
    df['notes'] = df['notes'].str.replace('\d+', '')
    all_docs = [
        'In June, an army soldier Aurangzeb of 44 Rashtriya Rifles posted in south Kashmir\x92s Shopian district was abducted by militants and his bullet-riddled body was found 10 kilometers away from the place of kidnapping. Soon after police came to know about abduction of the soldier, a joint police and army team was send to village and search operation was launched in the area. Officials said that Bhat was at his home at Qazipora when some unidentified gunmen abducted him from his house.']
    doc_clean = [clean(doc).split() for doc in df['notes']]
    # doc_clean = [' '.join(l) for l in doc_clean]
    # doc_clean = doc_clean[1:10]
    print(doc_clean)
    return doc_clean


def performLDA():
    doc_clean = cleandoc()
    # TODO: Check about LDA more and see how it works
    # doc_clean = "n June, an army soldier Aurangzeb of 44 Rashtriya Rifles posted in south Kashmir\x92s Shopian district was abducted by militants and his bullet-riddled body was found 10 kilometers away from the place of kidnapping.', 'Soon after police came to know about abduction of the soldier, a joint police and army team was send to village and search operation was launched in the area.', 'Officials said that Bhat was at his home at Qazipora when some unidentified gunmen abducted him from his house.".split()
    gensim_model = gensim.models.Word2Vec(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
    print(ldamodel.print_topics(num_topics=3, num_words=3))


# performLDA()


def generate_model(kernal, regularizer, gamma, training_data, training_target):
    svm_classifier = svm.SVC(kernel=kernal , C=regularizer, gamma=gamma)
    svm_classifier.fit(training_data, training_target)
    return svm_classifier


# model_svm = generate_model("rbf", 1, 0.01, doc_clean, data.mnist_training_target)

def create_heat_map(confusion_matrix):
    dataframe = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10,10))
    sb.heatmap(dataframe, annot=True, fmt="g", cbar=True)
    plt.show()

arr = np.array([[55,7],[8,14]])
create_heat_map(arr)


