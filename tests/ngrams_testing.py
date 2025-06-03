import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.similarity import jaccard_similarity, ngram_similarity, generate_ngrams

text1 = "the quick brown fox"
text2 = "quick brown fox jumps"

tokens1 = text1.split()
tokens2 = text2.split()

bigram_similarity = ngram_similarity(tokens1, tokens2, n=2)
trigram_similarity = ngram_similarity(tokens1, tokens2, n=3)

print(f"Bigram similarity: {bigram_similarity}")
print(f"Trigram similarity: {trigram_similarity}")
