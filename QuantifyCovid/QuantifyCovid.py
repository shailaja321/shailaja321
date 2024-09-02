from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
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

main = tkinter.Tk()
main.title("Quantifying COVID-19 Content in the Online Health Opinion War Using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
en_stop = set(nltk.corpus.stopwords.words('english'))
global text_data
global dictionary
global corpus
global ldamodel
global pro,anti

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

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="FacebookPost")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def processDataset():
    text.delete('1.0', END)
    global text_data
    text_data = []
    dataset = pd.read_csv(filename,encoding="ISO-8859-1")
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'Posts')
        clean = cleanPost(msg.strip('\n').strip().lower())
        clean = prepare_text_for_lda(clean)
        text_data.append(clean)
    text.insert(END,'Posts after processing\n\n')
    text.insert(END,str(text_data)+"\n\n")
                

def LDA():
    global dictionary
    global corpus
    global ldamodel
    text.delete('1.0', END)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    NUM_TOPICS = 30
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=6)
    text.insert(END,'LDA Extracted Topics\n\n')
    for topic in topics:
        text.insert(END,str(topic)+"\n")

def viewTopics():
    global pro,anti
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
    text.delete('1.0', END)
    text.insert(END,'Pro vaccines topics details\n\n')
    text.insert(END,str(pro)+"\n\n")
    text.insert(END,'Pro vaccines topics details\n\n')
    text.insert(END,str(anti))

    
def scoreGraph():
    pro_graph = []
    anti_graph = []
    for key in pro: 
        pro_graph.append(pro[key])
    for key in anti: 
        anti_graph.append(anti[key])
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Total Topics')
    plt.ylabel('Coherence scores')
    plt.plot(pro_graph, 'ro-', color = 'indigo')
    plt.plot(anti_graph, 'ro-', color = 'blue')
    plt.legend(['Pro-Vax', 'Anti-Vax'], loc='upper left')
    plt.title('Coherence Topic Scores Graph')
    plt.show()    
        
    
def graph():
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='mmds')
    #pyLDAvis.enable_notebook(local=True)
    pyLDAvis.show(lda_display)
    
font = ('times', 16, 'bold')
title = Label(main, text='Quantifying COVID-19 Content in the Online Health Opinion War Using Machine Learning')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Facebook Posts Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Process Dataset using Gensim & NLTK", command=processDataset, bg='#ffb3fe')
processButton.place(x=350,y=550)
processButton.config(font=font1) 

LDAforest = Button(main, text="Run LDA Topic Modelling to Extract Topics", command=LDA, bg='#ffb3fe')
LDAforest.place(x=750,y=550)
LDAforest.config(font=font1) 

topicButton = Button(main, text="View Pro & Anti Vaccines Topics", command=viewTopics, bg='#ffb3fe')
topicButton.place(x=50,y=600)
topicButton.config(font=font1) 

vaccine = Button(main, text="Pro & Anti Vaccine Graph", command=scoreGraph, bg='#ffb3fe')
vaccine.place(x=350,y=600)
vaccine.config(font=font1) 

graph = Button(main, text="pyLDAvis Topic Visualization", command=graph, bg='#ffb3fe')
graph.place(x=750,y=600)
graph.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
