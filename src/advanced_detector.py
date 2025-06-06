
from .preprocessing import preprocess_doc
from .similarity import (simple_word_overlap, 
                         tfidf_cosine_similarity,
                         ngram_similarity)

class AdvancedPlagiarismDetector:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                'word_overlap': 0.3,
                'tfidf_cosine': 0.4,
                'bigram_similarity': 0.2,
                'trigram_similarity': 0.1
            }
        
    def extract_features(self, doc1, doc2, corpus=None):
        """extract multiple similarity features between two documents"""
        tokens1 , tokens2 = preprocess_doc(doc1), preprocess_doc(doc2)
        if corpus is None:
            corpus = [doc1, doc2]
        features= {}
        
        # Word Overlap
        features["word_overlap"] = simple_word_overlap(tokens1, tokens2)
        # TF-IDF Cosine Similarity
        features["tfidf_cosine"] = tfidf_cosine_similarity(tokens1, tokens2, corpus)
        # N-gram Similarity
        features["bigram_similarity"] = ngram_similarity(tokens1, tokens2, n=2)
        features["trigram_similarity"] = ngram_similarity(tokens1, tokens2, n=3)
        
        #Length based Features
        len1, len2 = len(tokens1), len(tokens2)
        features["length_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        return features
    
    def calculate_weighted_similarity(self, features):
        """calculate weighted similarity score"""
        score = 0.0
        for feature, value in features.items():
            if feature in self.weights:
                score += self.weights[feature] * value
        return score
    
    def detect_plagiarism(self, doc1, doc2, threshold=0.5):
        """advanced plagiarism detection"""
        features = self.extract_features(doc1, doc2)
        weighted_score = self.calculate_weighted_similarity(features)
        result = {
            'similarity_score': weighted_score,
            'is_plagiarized': weighted_score >= threshold,
            'features': features,
            'threshold': threshold
        }
        
        return result
        