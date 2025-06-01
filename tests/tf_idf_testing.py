import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.similarity import calculate_tfidf, calculate_tf, calculate_idf

docs = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["the", "cat", "and", "the", "dog", "are", "friends"],
    ["the", "mat", "is", "on", "the", "floor"]
]
tfidf_results = calculate_tfidf(docs)
print("TF-IDF Results:", tfidf_results)