import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def load_web_icon_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    images = []
    alt_texts = []
    for json_path in tqdm(list(dataset_path.glob('*.json'))):
        with open(json_path, 'r') as f:
            attributes = json.load(f)['attributes']
        
        if attributes.get('alt'):
            image_path = json_path.parent / f'{json_path.name.split(".")[0]}.jpg'
            image = cv2.imread(str(image_path))
            images.append(image)
            
            alt_texts.append(attributes['alt'])
    
    return images, alt_texts
