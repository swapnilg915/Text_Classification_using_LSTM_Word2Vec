import os
import json

path = os.path.abspath(__file__)
BASE_PATH = os.path.dirname(path)
word2vec_model_path = os.path.join(BASE_PATH, "../../Embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz")
