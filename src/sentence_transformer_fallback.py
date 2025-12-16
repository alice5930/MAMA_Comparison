import hashlib
import numpy as np

class SentenceTransformer:
    def __init__(self, model_name: str = None):
        self._dim = 64

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, sentences, convert_to_numpy: bool = True):
        def _vec(s: str):
            h = hashlib.md5(str(s).encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            vec = np.resize(arr, self._dim)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-6)

        if isinstance(sentences, (list, tuple)):
            out = [_vec(x) for x in sentences]
            return np.stack(out) if convert_to_numpy else out
        return _vec(sentences)
