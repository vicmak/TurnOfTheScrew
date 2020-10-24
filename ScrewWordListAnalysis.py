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

#turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/subtitles.srt"
turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/turnofthescrew_text.txt"
chapters_folder = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/chapters/"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/cen300d.txt"
embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/word2vec.6B.300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/clmet.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/henry_james_300d.txt"




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

splitted_text = [word for word in book_text.split() if word not in stop_words]


# Words extracted by Victor from Wikipedia
#illness_words = ['hallucination', 'madness', 'sickness', 'illness', 'dream', 'confusion', 'psychosis', 'illusion']
#ghost_words = ['ghost', 'apparition', 'haunt', 'phantom', 'poltergeist', 'shade', 'specter', 'spirit', 'spook', 'wraith', 'soul']
#sexual_repression_words = ['sex', 'oppression', 'repression', 'control', 'honor', 'guilt', 'shame', 'marriage', 'rape']


# Words extracted by Yael from the novel Turn of the screw
#illness_words = ['fancies', 'fancy', 'fancied', 'anxious', 'nervous', 'nerves', 'shock', 'shaken', 'spell', 'sane', 'sanity', 'insane', 'exciting', 'distress', 'impression']
#ghost_words = ['visitation', 'visitant', 'visitor', 'strange', 'stranger', 'queer', 'apparition', 'monstrous', 'evil', 'unnatural']
#sexual_repression_words = ['passion', 'desire', 'naughty', 'free', 'innocence', 'intercourse', 'romance', 'romantic', 'infamous', 'corrupt', 'erect', 'pleasure', 'climax', 'love']


# Top k words extracted from embeddings space - James' bibliography
illness_words = ['artist', 'unconventional', 'imperturbable', 'ejaculation', 'omnibus', 'inexhaustible', 'unaffected', 'incurable', 'examination', 'unusually', 'illness', 'insane']
ghost_words = ['acceptance', 'indication', 'expectation', 'echo',  'coincidence', 'exaggeration', 'strangeness', 'excess', 'renewal', 'extension', 'evil', 'ghost']

# Top k words extracted from embeddings space - Wikipedia
#illness_words = ['mental', 'illnesses', 'ill', 'sick', 'diagnosed', 'suffering', 'insanity', 'ailment', 'disorder', 'debilitating']
#ghost_words = ['ghost', 'demon', 'beast', 'alien', 'creature', 'supernatural', 'haunted', 'mysterious', 'witch', 'demons', 'evil']

model = KeyedVectors.load_word2vec_format(embeddings_filepath)


print "GHOST WORDS:"

total_ghost_avg = 0.0

for ghost_word in ghost_words:
    sum_dist = 0.0
    for book_word in splitted_text:
        if book_word in model.vocab and ghost_word in model.vocab:
            current_word_distance = model.similarity(ghost_word, book_word)
            sum_dist = sum_dist + current_word_distance
    avg = float(sum_dist / len(splitted_text))
    print ghost_word, avg
    total_ghost_avg = total_ghost_avg + avg

print "TOTAL GHOST SCORE:", float(total_ghost_avg/len(ghost_words))

print "ILLNESS WORDS:"

total_illness_avg = 0.0

for illness_word in illness_words:
    sum_dist = 0.0
    for book_word in splitted_text:
        if book_word in model.vocab and illness_word in model.vocab:
            current_word_distance = model.similarity(illness_word, book_word)
            sum_dist = sum_dist + current_word_distance
    avg = float(sum_dist / len(splitted_text))
    print illness_word, avg
    total_illness_avg = total_illness_avg + avg

print "TOTAL ILLNESS SCORE:", float(total_illness_avg/len(illness_words))