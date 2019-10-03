import sys
import pickle

import cv2
import matplotlib.pyplot as plt

from src.auto_alt_text import preprocess_image


def main():
    inference_image_path, model_path = sys.argv[1:]

    print('loading image')
    inference_image = cv2.imread(inference_image_path)

    print('loading model')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print('getting alt text')
    alt_texts = model.get_alt_text(inference_image)

    print(alt_texts)


if __name__ == '__main__':
    main()
