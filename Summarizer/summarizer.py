import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize
import networkx as nx
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')

def read_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    sentences = []
    for s in df['article_text']:
        # s = s[:800]
        sentences.append(sent_tokenize(s))
    # sentences = [y for x in sentences for y in x]  # flatten list
    return sentences


# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def get_word_embeddings():
    pickle_in = open("Summarizer/glove.6B.100d.pkl", "rb")
    word_embeddings = pickle.load(pickle_in)
    return word_embeddings


def clean_sentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences


def get_sentence_vectors(clean_sentences, word_embeddings):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

def summarize_text(text):
    sentences = []
    sentences.append(sent_tokenize(text))


def summarize(content, isFile):
    sentences = []
    if isFile:
        sentences = read_data(content)
    else:
        sentences.append(sent_tokenize(content))

    word_embeddings = get_word_embeddings()
    output = []
    for item in sentences:
        cleaned_sentences = clean_sentences(item)
        sentence_vectors = get_sentence_vectors(cleaned_sentences, word_embeddings)

        # similarity matrix
        sim_mat = np.zeros([len(item), len(item)])
        for i in range(len(item)):
            for j in range(len(item)):
                if i != j:
                    sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[
                        0, 0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(item)), reverse=True)

        # Specify number of sentences to form the summary
        sn = 3

        # Generate summary
        result = []
        for i in range(sn):
            # print(ranked_sentences[i][1])
            if len(ranked_sentences) > i:
                result.append(ranked_sentences[i][1])
        output.append(result)

    return output







