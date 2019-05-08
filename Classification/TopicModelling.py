import DataPreparation.PreProcessData as ppd
import nltk
import gensim
from gensim import corpora


vac_data = ppd.read_data('../DataPreparation/2016-03-18-2019-03-18-OnlyVAC.csv')
vac_data_notes = vac_data['notes']


# TODO: Check about LDA more and see how it works
gensim_model = gensim.models.Word2Vec(vac_data_notes)
vac_data_notes_tokenized = [ nltk.word_tokenize(doc) for doc in vac_data_notes]
dictionary = corpora.Dictionary(vac_data_notes_tokenized)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in vac_data_notes_tokenized]
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))
