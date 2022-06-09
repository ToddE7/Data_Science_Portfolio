'''Manual implementation of averaged efficient perceptron for sentiment analysis NLP'''
'''Author: Todd Endres'''

from __future__ import division
import time
from svector import svector
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label == "+" else -1, words.split())

def cache_data(text):
    x = read_from(text)
    return x

def make_vector2(words): # prunes words and
    v = svector()
    word_counter = svector()
    for word in words:
        v[word] += 1
        word_counter[word] += 1
    return v, word_counter

def test(devfile, model):
    predictions = []
    pos = []
    neg = []
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):  # note 1...|D|
        pred = model.dot(make_vector(words))
        err += label * (model.dot(make_vector(words))) <= 0
        predictions.append((pred, words))
        if label == -1 and pred > 0: # add to lists where prediction is wrong
            neg.append((' '.join(words), pred))
        if label == 1 and pred < 0:
            pos.append((' '.join(words), pred))
    return err / i, predictions, pos, neg  # i is |D| now


def train2(trainfile, devfile, epochs=5):  # add bias and use averaged perceptron
    best_err = 1.
    w = svector()
    wa = svector()
    c = 0

    # setup train vector
    counter = svector()
    remove = svector()
    vals = []
    train_data = cache_data(trainfile)
    train_vec = []
    for i, (y, words) in enumerate(train_data):
        x = make_vector(words + ["<bias>"])
        train_vec.append([y, x])
        vals.append(x)
    # train_vec = [train_vec2[i] for i > 1] #svector({word: count for word, count in v.items() if count > 1})
    for (y, x) in train_vec:
        for word in x:
            counter[word] += x[word]
        remove = {word: count for word, count in counter.items() if count == 1}
        for word in x.copy():
            if word in remove:
                x.pop(word)

    # import dev/test data file
    dev_data = cache_data(devfile)
    dev_vec = []
    for i, (y, words) in enumerate(dev_data):
        x = make_vector2(words)
        dev_vec.append([y, x])

    # actual manual training process begins here:
    t = time.time()
    for it in range(1, epochs + 1):
        updates = 0
        dev_data = cache_data(devfile)
        for i, (y, x) in enumerate(train_vec, 1):  # label is +1 or -1
            if y * (w.dot(x)) <= 0:
                w += y * x
                wa += c * y * x
                updates += 1
            c += 1
        model_weight = w * c - wa
        dev_err, preds, pos, neg = test(devfile, model_weight)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return dev_err, preds, w, pos, neg # w is model weights, pos and neg are

def weights(model,n): # prints the highest and lowest weighted objects from the trained model
    high = sorted(model.items(), key=lambda x: x[1])[:n]
    low = sorted(model.items(), key=lambda x: x[1], reverse=True)[:n]

    print("Lowest feats:")
    for w in high:
        print(w)
    print("Highest feats:")
    for w in low:
        print(w)

# execute training
error2, preds2, model2, pos2, neg2 = train2("train.txt","dev.txt",10)

# convert model to an svector
mod = svector.convert(model2)

# get high and low weighted obj
#weights(model2,20)
#pos.sort(key = lambda i: i[1])
#neg.sort(key = lambda i: i[1], reverse=True)
#pos[:5]
#neg[:5]


