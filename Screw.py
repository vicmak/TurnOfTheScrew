
import numpy
# Set random seed to produce repeatable results
numpy.random.seed(7)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from keras.models import Sequential
import random
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
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
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace + punctuation)])


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
chapters_folder = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/chapters/"




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

ghost_words = ['ghost', 'apparition', 'haunt', 'phantom', 'poltergeist', 'shade', 'specter', 'spirit', 'spook', 'wraith', 'soul']
illness_words = ['hallucination', 'madness', 'sickness', 'illness', 'dream', 'confusion', 'psychosis', 'illusion']
scary_words = ['death', 'fear', 'afraid', 'creepy', 'scary']

model = KeyedVectors.load_word2vec_format('/Users/macbook/Desktop/corpora/embeddings/word2vec.6B.300d.txt')



def calc_semantic_distance(subject_words_list, text):
    splitted_text = [word for word in text.split() if word not in stop_words]
    total_avg = 0.0
    for subject_word in subject_words_list:
        sum_dist = 0.0
        for single_word in splitted_text:
            if single_word in model.vocab:
                single_word_distance = model.similarity(subject_word,single_word)
                sum_dist = sum_dist + single_word_distance
        total_avg = total_avg + float(sum_dist / len(splitted_text))
    return float(total_avg / len(subject_words_list))

print "CHAPTERS:"
for i in range(0, 25):
    chapter_file_name = chapters_folder+ str(i)+".txt"
    chapter_text = ""
    with open(chapter_file_name) as chapter_text_file:
        for line in chapter_text_file:
            line = separate_punctuation(line)
            line = line.lower()
            chapter_text = chapter_text + line
    chapter_text = ExtractAlphanumeric(chapter_text)
    print i, calc_semantic_distance(ghost_words, chapter_text), calc_semantic_distance(illness_words, chapter_text), calc_semantic_distance(scary_words, chapter_text)

print "GHOST WORDS:"

total_ghost_avg = 0.0

for ghost_word in ghost_words:
    sum_dist = 0.0
    for book_word in splitted_text:
        if book_word in model.vocab:
            current_word_distance = model.similarity(ghost_word, book_word)
            sum_dist = sum_dist + current_word_distance
    avg = float(sum_dist / len(splitted_text))
    #print ghost_word, avg
    total_ghost_avg = total_ghost_avg + avg

print "TOTAL GHOST SCORE:", float(total_ghost_avg/len(ghost_words))
print "SEMANTIC DIST:", calc_semantic_distance(ghost_words, book_text)

print "ILLNESS WORDS:"

total_illness_avg = 0.0

for illness_word in illness_words:
    sum_dist = 0.0
    for book_word in splitted_text:
        if book_word in model.vocab:
            current_word_distance = model.similarity(illness_word, book_word)
            sum_dist = sum_dist + current_word_distance
    avg = float(sum_dist / len(splitted_text))
    #print illness_word, avg
    total_illness_avg = total_illness_avg + avg

print "TOTAL ILLNESS SCORE:", float(total_illness_avg/len(illness_words))
print "SEMANTIC DIST:", calc_semantic_distance(illness_words, book_text)

def calc_semantic_distance2(subject_words_list, text):
    splitted_text = [word for word in text.split() if word not in stop_words]
    total_avg = 0.0
    for subject_word in subject_words_list:
        sum_dist = 0.0
        for single_word in splitted_text:
            if single_word in model.vocab:
                single_word_distance = model.similarity(subject_word,single_word)
                sum_dist = sum_dist + single_word_distance
        total_avg = total_avg + float(sum_dist / len(splitted_text))
    return float(total_avg / len(subject_words_list))


#freq_dist = nltk.FreqDist(splitted_text)



#print "FREQUENCY DISTRIBUTIONS:"
#print freq_dist.B()
#print freq_dist.N()
#print freq_dist.hapaxes()
#print freq_dist.most_common(50)
#print freq_dist["lie"]



#stopwords = set(STOPWORDS)
#wordcloud = WordCloud(width=800, height=800,
#                      background_color='white',
#                      stopwords=stopwords,
#                      min_font_size=10).generate(book_text)

# plot the WordCloud image
#plt.figure(figsize=(8, 8), facecolor=None)
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.tight_layout(pad=0)

#plt.show()