import math
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

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
