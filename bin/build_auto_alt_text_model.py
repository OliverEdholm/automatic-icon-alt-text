import sys
import pickle

from src.auto_alt_text import AutoAltText
from src.datasets import load_web_icon_dataset


def main():
    web_icon_dataset_path, model_save_path = sys.argv[1:]

    print('loading dataset')
    images, alt_texts = load_web_icon_dataset(web_icon_dataset_path)

    print('building model')
    auto_alt_text_model = AutoAltText(images, alt_texts)

    print('saving model')
    with open(model_save_path, 'wb') as f:
        pickle.dump(auto_alt_text_model, f)


if __name__ == '__main__':
    main()
