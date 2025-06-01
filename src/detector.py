from .preprocessing import preprocess_doc
from .similarity import simple_word_overlap


class BasicPlagiarismDetector:
    """basic plagiarism detector"""
    def __init__(self, threshold=0.5):
        if not isinstance(threshold, (int, float)):
            raise TypeError("Threshold must be a number")
        # Threshold should be between 0 and 1
        if not (0<= threshold <= 1):
            raise ValueError("Threshold must be obligatory between 0 and 1")
        self.threshold = threshold
        
    def detect(self, doc1, doc2):
        """detect plagiarism between two documents"""
        if doc1 is None or doc2 is None:
            raise ValueError("Both documents must be provided")
        if not isinstance(doc1, str) or not isinstance(doc2, str):
            raise TypeError("Both documents must be strings")
        if not doc1.strip() or not doc2.strip():
            raise ValueError("Both documents must be non-empty strings")
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
        if documents is None:
            raise ValueError("Documents list must be provided")
        if not hasattr(documents,'__iter__') or isinstance(documents, str):
            raise TypeError("Document should be an iterable, not a string (e.g. : list or tuple)")
        if len(documents) < 2:
            return ValueError("At least two documents are required for PlagAi Analysis")
        results = []
        n = len(documents)
        for i in range(n):
            for j in range(i+1,n):
                result = self.detect(documents[i], documents[j])
                result["doc1_index"] = i
                result["doc2_index"] = j
                results.append(result)
        return results
    
    