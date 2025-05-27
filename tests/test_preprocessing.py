import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_doc

sample_text = "Hello, world! This is a test document. It contains some punctuation, and stop words."
processed = preprocess_doc(sample_text)
print(f"Original text: {sample_text}")
print(f"Processed tokens: {processed}")
