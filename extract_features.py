import os
from os import listdir
from pickle import dump
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm

def extract_features(directory):
    # Load ResNet50 model
    # We remove the top classification layer to get the features from the CNN
    model = ResNet50()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())

    features = dict()
    print("Extracting features from images...")
    
    # Extract features for each image
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Please add dataset.")
        return None
        
    for name in tqdm(listdir(directory)):
        filename = directory + '/' + name
        try:
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            # Get image features
            feature = model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = feature
        except Exception as e:
            print(f"Skipping {filename}, error: {e}")
            
    return features

if __name__ == '__main__':
    # Define dataset directory, extracting and saving features using ResNet50
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, 'dataset', 'Images')
    
    if not os.path.exists(os.path.join(BASE_DIR, 'dataset')):
        os.makedirs(os.path.join(BASE_DIR, 'dataset'))
        
    print('Starting feature extraction...')
    features = extract_features(dataset_dir)
    
    if features:
        print(f'Extracted Features: {len(features)}')
        features_path = os.path.join(BASE_DIR, 'features.pkl')
        dump(features, open(features_path, 'wb'))
        print("Saved features to features.pkl")
    else:
        print("Please ensure the dataset is downloaded before running extract_features.py")
