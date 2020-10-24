from nltk.corpus import stopwords
from gensim.models import KeyedVectors

turn_of_the_screw_filename = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/turnofthescrew_text.txt"
other_book_filename = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/hallucinations/The Principles of Psychology william james.txt"


#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/cen300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/henry_james_300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/word2vec.6B.300d.txt"
embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/clmet.txt"

print "Reading embeddings..."
model = KeyedVectors.load_word2vec_format(embeddings_filepath)

stop_words = stopwords.words('english')

def extract_alphanumeric(ins):
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


def read_text(filename):
    book_text = ""
    with open(filename) as book_text_file:
        for line in book_text_file:
            line = separate_punctuation(line)
            line = line.lower()
            book_text = book_text + line
    return extract_alphanumeric(book_text)


def calc_wmd_distance(text1, text2):
    splitted_text1 = [word for word in text1.split() if word not in stop_words]
    splitted_text2 = [word for word in text2.split() if word not in stop_words]
    return model.wmdistance(splitted_text1, splitted_text2)

print "Reading book 1 ..."
turn_of_the_screw_text = read_text(turn_of_the_screw_filename)
print "Reading book 2 ..."
other_book_text = read_text(other_book_filename)
print "Calculating WMD..."
print "WMD:", calc_wmd_distance(turn_of_the_screw_text, other_book_text)

