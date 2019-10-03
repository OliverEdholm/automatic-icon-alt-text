from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from src.features import preprocess_image


class AutoAltText:
    def __init__(self, images, alt_texts):
        self._alt_texts = alt_texts

        preprocessed_images = [preprocess_image(image).flatten()
                               for image in tqdm(images)]
        
        self._pca = PCA(n_components=500)
        pca_features = self._pca.fit_transform(preprocessed_images)
        
        self._nearest_neighbors = NearestNeighbors(n_neighbors=10, metric='cosine').fit(
            pca_features)

    def get_alt_text(self, image):
        preprocessed_image = self._pca.transform([preprocess_image(image).flatten()])[0]

        alt_text_idxs = self._nearest_neighbors.kneighbors([preprocessed_image])[1][0]

        return [self._alt_texts[idx]
                for idx in alt_text_idxs]
