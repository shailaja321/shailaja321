import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import pickle
import gensim
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict

en_stop = set(nltk.corpus.stopwords.words('english'))

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    #print(tokens)
    return tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)    

def prepare_text_for_lda(text):
    tokens = text.split(" ")
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

text_data = []
dataset = pd.read_csv('FacebookPost/posts.csv',encoding="ISO-8859-1")
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'Posts')
    clean = cleanPost(msg.strip('\n').strip().lower())
    clean = prepare_text_for_lda(clean)
    text_data.append(clean)

print(text_data)
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
NUM_TOPICS = len(dataset)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='mmds')
#pyLDAvis.enable_notebook(local=True)
#pyLDAvis.show(lda_display)
anti_topics = ['shot','burder','protest','avoid','flu','fake','stop','afraid','never','test','spread','poison']
pro_topics = ['maskwearing','protect','healthcare','trust','ailment','mask','wash','distancing','distance','soap','prevent','mandatory']
pro =  {}
anti = {}
combine = {}
for i in range(len(text_data)):
    data = text_data[i]
    for j in range(len(data)):
        if data[j] in anti_topics:
            if data[j] in anti:
                anti[data[j]] = anti.get(data[j]) + 1
            else:
                anti[data[j]] = 1
        if data[j] in pro_topics:
            if data[j] in pro:
                pro[data[j]] = pro.get(data[j]) + 1
            else:
                pro[data[j]] = 1        
                
print(pro)
print("===============")
print(anti)


