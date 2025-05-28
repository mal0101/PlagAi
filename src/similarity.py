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