from .preprocessing import preprocess_doc
from .similarity import simple_word_overlap

class BasicPlagiarismDetector:
    """basic plagiarism detector"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def detect(self, doc1, doc2):
        """detect plagiarism between two documents"""
        tokens1 = preprocess_doc(doc1)
        tokens2 = preprocess_doc(doc2)
        
        similarity = simple_word_overlap(tokens1, tokens2)
        
        result = {
            "similarity": similarity,
            "is_plagiarized": similarity >= self.threshold,
            "threshold": self.threshold
        }
        
        return result
    
    def analyze_documents(self, documents):
        """anamyzes multiple documents for plagiarism"""
        results = []
        n = len(documents)
        for i in range(n):
            for j in range(i+1,n):
                result = self.detect(documents[i], documents[j])
                result["doc1_index"] = i
                result["doc2_index"] = j
                results.append(result)
        return results
    
    