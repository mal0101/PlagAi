import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import BasicPlagiarismDetector

detector = BasicPlagiarismDetector(threshold=0.1)

doc1 = "Artificial intelligence is transforming the world quickly."
doc2 = "AI is changing the world significantly and rapidly."
doc3 = "Machine learning algorithms are very powerful"

#Test two documents 
result = detector.detect(doc1, doc2)
print(f"Result: {result}")
#Test multiple documents
docs = [doc1, doc2, doc3]
results = detector.analyze_documents(docs)
for res in results:
    print(f"Doc1 Index: {res['doc1_index']}, Doc2 Index: {res['doc2_index']}, Similarity: {res['similarity']:.2f}, Plagiarized: {res['is_plagiarized']}")
