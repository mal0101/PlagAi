import math
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity(d1,d2):
    """Calculate Jaccard similairty between two docs"""
    intersection = d1.intersection(d2)
    union = d1.union(d2)
    if not union:
        return 0.0
    return len(intersection)/len(union)

def overlap_coefficient(d1,d2):
    """claculate the overlap coefficient between two docs"""
    intersection = d1.intersection(d2)
    min_size = min(len(d1), len(d2))
    if min_size == 0:
        return 0.0
    return len(intersection) / min_size


def simple_word_overlap(tokens1,tokens2):
    """calculate simple word overlap percentage"""
    set1, set2 = set(tokens1), set(tokens2)
    return jaccard_similarity(set1, set2)

def calculate_tf(tokens):
    """calculate term frequency for a list of tokens"""
    tf_dict = {}
    total_tokens =len(tokens)
    tokens_count = Counter(tokens)
    for token, count in tokens_count.items():
        tf_dict[token] = count / total_tokens
        
    return tf_dict

def calculate_idf(documents):
    """calculate inverse document frequency for a list of documents"""
    idf_dict = {}
    total_docs = len(documents)
    all_tokens = set(token for doc in documents for token in doc)
    for token in all_tokens:
        containing_docs = sum(1 for doc in documents if token in doc)
        idf_dict[token] = math.log(total_docs / containing_docs) if containing_docs > 0 else 0.0 # the if condition prevents division by zero
        
    return idf_dict

def calculate_tfidf(documents):
    """calculate TF-IDF for a list of documents"""
    idf = calculate_idf(documents)
    tfidf_docs = []
    for doc in documents: 
        tf = calculate_tf(doc)
        tfidf = {}
        for token in doc:
            tfidf[token] = tf[token] * idf[token]
            
        tfidf_docs.append(tfidf)
    return tfidf_docs

def vector_cosine_similarity(vec1, vec2):
    """calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def tfidf_cosine_similarity(doc1_tokens, doc2_tokens, corpus_tokens):
    """calculate cosine similarity using TF-IDF vectors"""
    # Create vocabulary
    vocab = list(set(doc1_tokens + doc2_tokens + [t for doc in corpus_tokens for t in doc]))
    # Create documents to TF-IDF vectors
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    # Prepare documents as strings (rejoined tokens)
    doc1_str = ' '.join(doc1_tokens)
    doc2_str = ' '.join(doc2_tokens)
    tf_idf_matrix = vectorizer.fit_transform([doc1_str, doc2_str])
    # Calculate cosine similarity
    similarity = cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix[1:2])
    
    return similarity
