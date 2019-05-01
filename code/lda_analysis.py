import csv
import unicodedata
import string
import csv
import random


category_lines_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
category_lines = {}
all_categories = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
n_categories = len(all_categories)
def read_lines():
	with open('../data/test_data.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		print("YO")
		for row in csv_reader:
			if (line_count != 0):
				text = row[1]
				category_lines_list[int(row[0])].append(text)
			line_count += 1

		for i in range(0,16):
			category_lines[str(i)] = category_lines_list[i]
# Build the category_lines dictionary, a list of lines per category
lines = read_lines()
corpus = [item for sublist in category_lines_list for item in sublist]

import gensim
from gensim import corpora, models
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
nltk.download('wordnet')
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = [preprocess(i) for i in corpus]


import gensim
from gensim import corpora, models
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.25, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

for i in range(0, 16):
	docs = category_lines[str(i)][0:5]
	print(i)
	print(docs)
	for j in docs:
		v = tfidf[j]
		vect = lda_model_tfidf[v]
		print(vect)
	print("\n")

