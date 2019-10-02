from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from src.features import get_hog_feature


class AutoAltText:
    def __init__(self, images, alt_texts):
        self._alt_texts = alt_texts

        hog_features = [get_hog_feature(image)
                        for image in tqdm(images)]
        self._nearest_neighbors = NearestNeighbors(n_neighbors=1, metric='cosine').fit(
            hog_features)

    def get_alt_text(self, image):
        hog_feature = get_hog_feature(image)

        alt_text_idx = self._nearest_neighbors.kneighbors([hog_feature])[1][0][0]

        return self._alt_texts[alt_text_idx]
 