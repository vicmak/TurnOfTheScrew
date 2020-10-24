import math
import nltk
import os
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def single_KL(p, q):
    if p*q == 0:
        return 0
    return p*math.log(p/q)


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace)])


def read_files(path):

    all_text_content = ""
    for root, dirs, files in os.walk(path):
        for filename in files:
            full_file_path = root + "/" + filename
            with open(full_file_path) as f:
                content = f.read()
                all_text_content = all_text_content + " " + content.lower()

    return ExtractAlphanumeric(all_text_content)


topic1_path = "/Users/macbook/Desktop/corpora/TurnOfTheScrew/book_text"
#topic2_path = "/Users/macbook/Desktop/corpora/clmet/corpus/txt/plain"
topic2_path = "/Users/macbook/Desktop/corpora/clmet/corpus/txt/plain"
#topic2_path = "/Users/macbook/Desktop/corpora/cen/cen"
print "Reading topic 1"
topic1_text = read_files(topic1_path)
print "Reading topic 2"
topic2_text = read_files(topic2_path)

topic1 = 'The Book: Turn of the screw'
topic2 = 'Corpus of English Novels'

titles_tokens = {topic1: {}, topic2: {}}
ignmore_symbols = [".", ":", ",", "'s", "'"]

print "Tokenizing topic 1"
topic1_tokens = nltk.word_tokenize(topic1_text)
print "Tokenizing topic 2"
topic2_tokens = nltk.word_tokenize(topic2_text)

print "Calculating topic 1 tokens"
for t in topic1_tokens:
    if t not in stop_words:
        if t not in titles_tokens[topic1]:
            titles_tokens[topic1][t] = 1
            titles_tokens[topic2][t] = 1
        titles_tokens[topic1][t] += 1

print "Calculating topic 2 tokens"
for t in topic2_tokens:
    if t not in stop_words:
        if t not in titles_tokens[topic2]:
            titles_tokens[topic2][t] = 1
        if t not in titles_tokens[topic1]:
            titles_tokens[topic1][t] = 1
        titles_tokens[topic2][t] += 1

sum1 = sum(titles_tokens[topic1].values())
sum2 = sum(titles_tokens[topic2].values())
print sum1, sum2

kl_tokens = {}
topk = 60
for t in titles_tokens[topic1]:
    p1 = float(float(titles_tokens[topic1][t])/float(sum1))
    p2 = float(float(titles_tokens[topic2][t])/float(sum2))
    kl1 = single_KL(p1, p2)
    kl2 = single_KL(p2, p1)
    kl_tokens[t] = {topic1:kl1, topic2:kl2}

print "KL ", topic1, "to", topic2
for t in sorted(kl_tokens, key=lambda x: kl_tokens[x][topic1], reverse=True)[0:topk]:
    print t, "\t", kl_tokens[t][topic1], "\t",  kl_tokens[t][topic2]
print
print "--------"
print "   "
print "KL ", topic2, "to", topic1
for t in sorted(kl_tokens, key=lambda x: kl_tokens[x][topic2], reverse=True)[0:topk]:
    print t, "\t", kl_tokens[t][topic1], "\t",  kl_tokens[t][topic2]




