import sys
import pickle

import cv2


def main():
    inference_image_path, model_path = sys.argv[1:]

    print('loading image')
    inference_image = cv2.imread(inference_image_path)

    print('loading model')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print('getting alt text')
    print(model.get_alt_text(inference_image))


if __name__ == '__main__':
    main()
