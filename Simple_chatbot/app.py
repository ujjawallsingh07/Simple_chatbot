# app.py
from flask import Flask, render_template, request, session, redirect, url_for
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from datetime import timedelta

# ensure punkt is available
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'replace-with-a-strong-secret'  # change for production
app.permanent_session_lifetime = timedelta(days=1)

stemmer = PorterStemmer()

# Load model & assets
model = load_model('model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json', 'r', encoding='utf-8'))

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words_list):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': float(r[1])})
    return return_list

def get_response(ints):
    if not ints:
        return "I didn't understand that. Can you try rephrasing?"
    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Hmm, I have nothing for that."

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize session chat history
    if 'chat' not in session:
        session['chat'] = []
    if request.method == 'POST':
        user_msg = request.form.get('message', '').strip()
        if user_msg:
            # Save user message
            session['chat'].append({'who': 'user', 'text': user_msg})
            # Predict & reply
            ints = predict_class(user_msg)
            bot_reply = get_response(ints)
            session['chat'].append({'who': 'bot', 'text': bot_reply})
            session.modified = True  # tell Flask session changed
        return redirect(url_for('home'))
    return render_template('index.html', chat=session.get('chat', []))

@app.route('/clear')
def clear():
    session.pop('chat', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
