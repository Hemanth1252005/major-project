from numpy import array
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import load_set, load_clean_descriptions, create_tokenizer, max_length
from model import define_model

# Create data generator, intended to be used in a call to model.fit()
def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            if key not in features:
                continue
            feature = features[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, feature, vocab_size)
            yield (in_img, in_seq), out_word

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

if __name__ == '__main__':
    # load photo features
    # features.pkl is generated using extract_features.py
    if not os.path.exists('features.pkl'):
        print(f"File features.pkl not found. Run extract_features.py first.")
        exit(1)
        
    all_features = pickle.load(open('features.pkl', 'rb'))
    train = set(all_features.keys())
    print(f'Extracted Features / Images: {len(train)}')

    # load clean train descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print(f'Descriptions: train={len(train_descriptions)}')

    train_features = {k: all_features[k] for k in train if k in all_features and k in train_descriptions}
    print(f'Photos: train={len(train_features)}')

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f'Vocabulary Size: {vocab_size}')
    
    # save the tokenizer
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print(f'Description Length: {max_length}')

    # define the model
    model = define_model(vocab_size, max_length)

    # train the model, run epochs manually and save after each epoch
    epochs = 20
    steps = len(train_descriptions)
    if not os.path.exists('models'):
        os.makedirs('models')
        
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
        print(f'Epoch {i+1}/{epochs}')
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model.save(f'models/model_{i}.h5')
        
    print("Training finished. Final output saved to models/")
