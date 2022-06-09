'''Sentiment Analysis NLP with Tensor Flow LSTM'''
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

def remover(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

def counter_word(text):
    count = Counter()
    for text in text.values:
        for word in text.split():
            count[word] += 1
    return count

sentences = []
labels = []
for i, (label, words) in enumerate(read_from("train.txt"), 1): # import training data
    sentences.append(words)
    labels.append(label)

labels = pd.DataFrame(labels, columns = ['target'])

sentences = pd.DataFrame(sentences)
sentences = sentences.applymap(str)
cols = sentences.columns

final = pd.DataFrame()
final['combined'] = sentences[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

stop = set(stopwords.words("english")) # remove stop words
stop.add("none")
stop.add(",")

final["combined"] = final.combined.map(remover)

dev_sentences = []
dev_labels = []
for i, (label, words) in enumerate(read_from("dev.txt"), 1): # import dev/test data
    dev_sentences.append(words)
    dev_labels.append(label)

dev_labels = pd.DataFrame(dev_labels, columns = ['target'])

dev_sentences = pd.DataFrame(dev_sentences)
dev_sentences = dev_sentences.applymap(str)
dev_cols = dev_sentences.columns

dev = pd.DataFrame()
dev['dev_combined'] = dev_sentences[dev_sentences.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

counter = counter_word(final.combined)
unique_words = len(counter)

train_sentences = final.combined.to_numpy() # setup data for modeling
train_labels = labels.target.to_numpy()
val_sentences = dev.dev_combined.to_numpy()
val_labels = dev_labels.target.to_numpy()

train_labels = np.where(train_labels < 0, 0, train_labels)
val_labels = np.where(val_labels < 0, 0, val_labels)

'''Tokenize data for modeling'''
tokenizer = Tokenizer(num_words=unique_words)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

max_len = 30

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding="post", truncating="post")

reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])

def decode(seq):
    return " ".join([reverse_word_index.get(idx, "?") for idx in seq])

'''Begin setting up tensorflow keras model'''

model = keras.models.Sequential()
model.add(layers.Embedding(unique_words, 32, input_length=max_len))

model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

#print(model.summary()) # if you want to print the model summary to look at it

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=.001)
metric = ["Accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metric)

model.fit(train_padded, train_labels, epochs=10, validation_data=(val_padded, val_labels), verbose=2)

'''Begin importing test data and predicting'''

test_sentences = []
test_labels = []
for i, (label, words) in enumerate(read_from("test.txt"), 1):
    test_sentences.append(words)
    test_labels.append(label)

test_sentences = pd.DataFrame(test_sentences)
test_sentences = test_sentences.applymap(str)
test_cols = test_sentences.columns

test = pd.DataFrame()
test['combined'] = test_sentences[test_sentences.columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

test["combined"] = test.combined.map(remover)

test_sentences = test.combined.to_numpy()
test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, maxlen=max_len, padding="post", truncating="post")

predictions = model.predict(test_padded)
predictions = ["+" if p > 0.5 else "-" for p in predictions]

'''Write predictions to file in the same format as train and dev'''

with open('test.txt.predicted', 'w') as wf:
    for i, line in enumerate(open("test.txt")):
        label, words = line.strip().split("\t")
        print(f'{predictions[i]}\t{words}', file=wf)
