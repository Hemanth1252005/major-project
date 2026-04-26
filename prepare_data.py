import os
from utils import save_descriptions, clean_descriptions, to_vocabulary

def prepare_dataset():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_text = os.path.join(BASE_DIR, 'dataset', 'captions.txt')
    
    if not os.path.exists(dataset_text):
        print(f"Error: {dataset_text} not found.")
        return
        
    print("Loading and parsing descriptions...")
    descriptions = dict()
    with open(dataset_text, 'r', encoding='utf-8') as file:
        next(file) # skip header
        for line in file:
            tokens = line.strip().split(',', 1)
            if len(tokens) < 2:
                continue
            image_id, image_desc = tokens[0], tokens[1]
            image_id = image_id.split('.')[0]
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
            
    print(f"Loaded: {len(descriptions)} descriptions.")
    
    print("Cleaning descriptions...")
    clean_descriptions(descriptions)
    
    print("Building vocabulary...")
    vocabulary = to_vocabulary(descriptions)
    print(f"Vocabulary Size: {len(vocabulary)}")
    
    print("Saving cleaned descriptions to descriptions.txt...")
    save_path = os.path.join(BASE_DIR, 'descriptions.txt')
    save_descriptions(descriptions, save_path)
    print("Done! You can now proceed to run train.py")

if __name__ == '__main__':
    prepare_dataset()
