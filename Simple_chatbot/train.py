# train.py
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download once (will create nltk data folder)
nltk.download('punkt')

stemmer = PorterStemmer()

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0].astype(object)))
train_y = np.array(list(training[:, 1].astype(object)))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

print("Training model...")
model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

model.save('model.h5')
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Training finished. Saved model.h5, words.pkl, classes.pkl")
