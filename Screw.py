
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
embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/henry_james_300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/word2vec.6B.300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/clmet.txt"

book_text = ""

model = KeyedVectors.load_word2vec_format(embeddings_filepath)


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

#scary_words = ['fear', 'afraid', 'creepy', 'scary', 'danger', 'phobia', 'threat', 'horror', 'anxiety']


# Words extracted by Victor from Wikipedia
#illness_words = ['hallucination', 'madness', 'sickness', 'illness', 'dream', 'confusion', 'psychosis', 'illusion']
#ghost_words = ['ghost', 'apparition', 'haunt', 'phantom', 'poltergeist', 'shade', 'specter', 'spirit', 'spook', 'wraith', 'soul']
#sexual_repression_words = ['sex', 'oppression', 'repression', 'control', 'honor', 'guilt', 'shame', 'marriage', 'rape']


# Words extracted by Yael from the novel Turn of the screw
illness_words = ['fancies', 'fancy', 'fancied', 'anxious', 'nervous', 'nerves', 'shock', 'shaken', 'spell', 'sane', 'sanity', 'insane', 'exciting', 'distress', 'impression']
ghost_words = ['visitation', 'visitant', 'visitor', 'strange', 'stranger', 'queer', 'apparition', 'monstrous', 'evil', 'unnatural']
#sexual_repression_words = ['passion', 'desire', 'naughty', 'free', 'innocence', 'intercourse', 'romance', 'romantic', 'infamous', 'corrupt', 'erect', 'pleasure', 'climax', 'love']


# Top k words extracted from embeddings space - James' bibliography
#illness_words = ['artist', 'unconventional', 'imperturbable', 'ejaculation', 'omnibus', 'inexhaustible', 'unaffected', 'incurable', 'examination', 'unusually', 'illness', 'insane']
#ghost_words = ['acceptance', 'indication', 'expectation', 'echo',  'coincidence', 'exaggeration', 'strangeness', 'excess', 'renewal', 'extension', 'evil', 'ghost']

# Top k words extracted from embeddings space - Wikipedia
#illness_words = ['mental', 'illnesses', 'ill', 'sick', 'diagnosed', 'suffering', 'insanity', 'ailment', 'disorder', 'debilitating']
#ghost_words = ['ghost', 'demon', 'beast', 'alien', 'creature', 'supernatural', 'haunted', 'mysterious', 'witch', 'demons', 'evil']



def calc_semantic_distance(subject_words_list, text):
    splitted_text = [word for word in text.split() if word not in stop_words]
    total_avg = 0.0
    for subject_word in subject_words_list:
        sum_dist = 0.0
        for single_word in splitted_text:
            if single_word in model.vocab and subject_word in model.vocab:
                single_word_distance = model.similarity(subject_word,single_word)
                sum_dist = sum_dist + single_word_distance
        total_avg = total_avg + float(sum_dist / len(splitted_text))
    return float(total_avg / len(subject_words_list))


def calc_wmd_distance(subject_words_list, text):
    splitted_text = [word for word in text.split() if word not in stop_words]
    return model.wmdistance(subject_words_list, splitted_text)


def analyze_chapters_punctuation():
    print "Punctuation Analysis:"
    current_number_commas = 0.0
    current_number_periods = 0.0
    for i in range(0, 25):
        chapter_file_name = chapters_folder+ str(i)+".txt"
        chapter_text = ""
        with open(chapter_file_name) as chapter_text_file:
            for line in chapter_text_file:
                line = separate_punctuation(line)
                line = line.lower()
                chapter_text = chapter_text + line
        current_number_commas = current_number_commas + chapter_text.count(',')
        current_number_periods = current_number_periods + chapter_text.count('.')
        print i, float(current_number_commas / current_number_periods)


def analyze_chapters_timeline():
    print "Chapter Analysis:"
    for i in range(0, 25):
        chapter_file_name = chapters_folder+ str(i)+".txt"
        chapter_text = ""
        with open(chapter_file_name) as chapter_text_file:
            for line in chapter_text_file:
                line = separate_punctuation(line)
                line = line.lower()
                chapter_text = chapter_text + line
        chapter_text = ExtractAlphanumeric(chapter_text)
        print i, calc_wmd_distance(illness_words, chapter_text), calc_wmd_distance(ghost_words, chapter_text)


def analyze_publication_timeline():
    print "Publication Analysis:"
    for i in range(1, 24, 2):
        chapter_file_name = chapters_folder + str(i)+".txt"
        chapter2_file_name = chapters_folder + str(i+1)+".txt"
        chapter_text = ""
        with open(chapter_file_name) as chapter_text_file:
            for line in chapter_text_file:
                line = separate_punctuation(line)
                line = line.lower()
                chapter_text = chapter_text + line
        chapter_text = ExtractAlphanumeric(chapter_text)

        chapter2_text = ""
        with open(chapter2_file_name) as chapter2_text_file:
            for line in chapter2_text_file:
                line = separate_punctuation(line)
                line = line.lower()
                chapter2_text = chapter2_text + line
        chapter2_text = ExtractAlphanumeric(chapter2_text)

        chapter_text = chapter_text + chapter2_text

        print i, calc_semantic_distance(illness_words, chapter_text), calc_semantic_distance(ghost_words, chapter_text)

analyze_publication_timeline()

#analyze_chapters_punctuation()

#analyze_chapters_timeline()

#print "ILLNESS SEMANTIC RELATEDNESS:", calc_semantic_distance(illness_words, book_text)
#print "GHOST SEMANTIC RELATEDNESS:", calc_semantic_distance(ghost_words, book_text)
#print "SCARY SEMANTIC RELATEDNESS:", calc_semantic_distance(scary_words, book_text)
#print "REPRESSION SEMANTIC RELATEDNESS:", calc_semantic_distance(sexual_repression_words, book_text)

#print "WMD ILL", model.wmdistance(illness_words, splitted_text)
#print "WMD GHOST", model.wmdistance(ghost_words, splitted_text)
#print "WMD Scary", model.wmdistance(scary_words, splitted_text)
#print "WMD REPRESSION", model.wmdistance(sexual_repression_words, splitted_text)




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