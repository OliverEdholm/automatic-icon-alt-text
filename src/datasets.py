import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def load_web_icon_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    images = []
    alt_texts = []
    for json_path in tqdm(list(dataset_path.glob('*.json'))):
        try:
            with open(str(json_path), 'r') as f:
                attributes = json.load(f)['attributes']

            if attributes.get('alt'):
                image_path = str(json_path.parent) + '/' + '{}.jpg'.format(json_path.name.split(".")[0])
                image = cv2.imread(str(image_path))
                
                if image is not None:
                    images.append(image)

                    alt_texts.append(attributes['alt'])
        except Exception as e:
            pass
    
    return images, alt_texts
