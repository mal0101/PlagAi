import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity import simple_word_overlap
from src.preprocessing import preprocess_doc

doc1 = "The quick brown fox jumps over the lazy dog."
doc2 = "A quick brown fox leaps over a lazy dog."

tokens1 = preprocess_doc(doc1)
tokens2 = preprocess_doc(doc2)

similarity = simple_word_overlap(tokens1, tokens2)
print(f"Simple Word Overlap Similarity: {similarity:.2f}")
