import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import define_model

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
max_length = 34
vocab_size = len(tokenizer.word_index) + 1
model = define_model(vocab_size, max_length)
model.load_weights('models/model_19.h5')

# Dummy random feature
feature = np.random.rand(1, 2048).astype('float32')

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

in_text = 'startseq'
for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([feature, sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = word_for_id(yhat, tokenizer)
    if word is None:
        break
    in_text += ' ' + word
    if word == 'endseq':
        break
print("Caption:", in_text)
