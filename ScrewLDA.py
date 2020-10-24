
import numpy
# Set random seed to produce repeatable results
numpy.random.seed(7)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
#from keras.models import Sequential
import random
#from keras.layers import Dense
#from keras.layers import LSTM, Bidirectional
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import KeyedVectors
import mmap
import os
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace)])


def separate_punctuation(line):
    line = line.replace(".", " .")
    line = line.replace(",", " ,")
    line = line.replace("?", " ?")
    line = line.replace("!", " !")
    line = line.replace(":", " :")
    line = line.replace(";", " ;")
    line = line.replace("_", " ")

    return line

turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/turnofthescrew_text.txt"
#turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/subtitles.txt"

chapters_folder = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/chapters/"



def get_chapters_timeline():
    print "CHAPTERS:"
    chapters = []
    for i in range(0, 25):
        chapter_file_name = chapters_folder+ str(i)+".txt"
        chapter_text = ""
        with open(chapter_file_name) as chapter_text_file:
            for line in chapter_text_file:
                line = separate_punctuation(line)
                line = line.lower()
                chapter_text = chapter_text + line
        chapter_text = ExtractAlphanumeric(chapter_text)
        chapters.append([word.encode() for word in chapter_text.split() if word not in stop_words])
    return chapters

book_text = ""


with open(turn_of_the_screw_text) as book_text_file:
    for line in book_text_file:
        line = separate_punctuation(line)
        line = line.lower()
        book_text = book_text + line


book_text = ExtractAlphanumeric(book_text)

stop_words = stopwords.words('english')
# Show stop words
print "STOP WORDS:",stop_words[:5]

splitted_text = [word.encode() for word in book_text.split() if word not in stop_words]

ghost_words = ['ghost', 'apparition', 'haunt', 'phantom', 'poltergeist', 'shade', 'specter', 'spirit', 'spook', 'wraith', 'soul']
illness_words = ['hallucination', 'madness', 'sickness', 'illness', 'dream', 'confusion', 'psychosis', 'illusion']
scary_words = ['death', 'fear', 'afraid', 'creepy', 'scary']

print splitted_text
chapters = get_chapters_timeline()
#splitted_text = ['a','b']
dictionary_of_words = gensim.corpora.Dictionary(chapters)
print dictionary_of_words
word_corpus = [dictionary_of_words.doc2bow(word) for word in chapters]

lda_model = gensim.models.ldamodel.LdaModel(corpus=word_corpus,
                                            id2word=dictionary_of_words,
                                            num_topics=20,
                                            random_state=101,
                                            update_every=1,
                                            chunksize=30,
                                            passes=100,
                                            alpha='auto',
                                            per_word_topics=True)

coherence_val = gensim.models.CoherenceModel(model=lda_model, texts=chapters, dictionary=dictionary_of_words, coherence='c_v').get_coherence()

print('Coherence Score: ', coherence_val)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))