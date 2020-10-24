import gensim
from gensim.models import KeyedVectors


turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/subtitles.srt"
#turn_of_the_screw_text = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/turnofthescrew_text.txt"
chapters_folder = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/chapters/"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/cen300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/henry_james_300d.txt"
embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/word2vec.6B.300d.txt"
#embeddings_filepath = "/Users/macbook/Desktop/corpora/embeddings/clmet.txt"

model = KeyedVectors.load_word2vec_format(embeddings_filepath)

#print model.most_similar(positive=['ghost', 'visitant', 'apparition', 'evil'])

print model.most_similar(positive=['ghost', 'evil'])

print model.most_similar(positive=['insane', 'illness'])



