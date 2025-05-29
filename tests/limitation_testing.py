import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import BasicPlagiarismDetector
from src.preprocessing import preprocess_doc
from src.similarity import simple_word_overlap

def test_edge_cases():
    """Test various edge cases that can break the current implementation"""
    
    print("=" * 60)
    print("TESTING LIMITATIONS OF PLAGIARISM DETECTION SYSTEM")
    print("=" * 60)
    
    detector = BasicPlagiarismDetector(threshold=0.5)
    
    # Test cases that will reveal limitations
    test_cases = [
        # Case 1: Empty and None inputs
        {
            "name": "Empty/None Inputs",
            "docs": ["", None, "   ", "Normal text here"],
            "expected_issues": "NoneType errors, empty string handling"
        },
        
        # Case 2: Non-string inputs
        {
            "name": "Non-string Inputs", 
            "docs": [123, ["list", "input"], {"dict": "input"}, "Normal text"],
            "expected_issues": "AttributeError on non-string types"
        },
        
        # Case 3: Very large documents
        {
            "name": "Large Documents",
            "docs": ["word " * 10000, "word " * 10000, "different content"],
            "expected_issues": "Performance degradation, memory usage"
        },
        
        # Case 4: Special characters and encoding
        {
            "name": "Special Characters",
            "docs": ["Héllo wörld! 中文测试", "émojis 🚀🎉", "normal text"],
            "expected_issues": "Unicode handling, tokenization issues"
        },
        
        # Case 5: Only punctuation/numbers
        {
            "name": "Only Punctuation/Numbers",
            "docs": ["!!!", "123 456 789", "... --- ___"],
            "expected_issues": "Empty token lists after preprocessing"
        },
        
        # Case 6: Paraphrased content (semantic similarity)
        {
            "name": "Semantic Plagiarism",
            "docs": [
                "The cat sat on the mat",
                "A feline rested upon the rug", 
                "Dogs are great pets"
            ],
            "expected_issues": "Fails to detect semantic similarity"
        },
        
        # Case 7: Word order changes
        {
            "name": "Word Order Changes",
            "docs": [
                "Machine learning algorithms are powerful tools",
                "Powerful tools are machine learning algorithms",
                "Completely different content here"
            ],
            "expected_issues": "May not detect structural plagiarism"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"Expected Issues: {test_case['expected_issues']}")
        print("-" * 40)
        
        try:
            # Test individual document processing
            for j, doc in enumerate(test_case['docs']):
                try:
                    print(f"Processing doc {j}: {repr(doc)[:50]}...")
                    if doc is not None:
                        tokens = preprocess_doc(doc)
                        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                    else:
                        print(f"  Doc is None - this will cause issues")
                except Exception as e:
                    print(f"  ERROR in preprocessing: {type(e).__name__}: {e}")
            
            # Test pairwise detection
            print("\nTesting pairwise detection:")
            results = detector.analyze_documents(test_case['docs'])
            for result in results:
                print(f"  Docs {result['doc1_index']}-{result['doc2_index']}: "
                      f"Similarity={result['similarity']:.3f}, "
                      f"Plagiarized={result['is_plagiarized']}")
                      
        except Exception as e:
            print(f"CRITICAL ERROR: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE AND SCALABILITY TESTS")
    print("=" * 60)
    
    # Test performance with increasing document sizes
    sizes = [100, 1000, 5000]
    for size in sizes:
        print(f"\nTesting with {size} words per document:")
        try:
            import time
            large_doc1 = "word " * size
            large_doc2 = "word " * (size // 2) + "different " * (size // 2)
            
            start_time = time.time()
            result = detector.detect(large_doc1, large_doc2)
            end_time = time.time()
            
            print(f"  Time taken: {end_time - start_time:.3f} seconds")
            print(f"  Similarity: {result['similarity']:.3f}")
            print(f"  Memory usage: ~{sys.getsizeof(large_doc1) + sys.getsizeof(large_doc2)} bytes")
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

def test_threshold_edge_cases():
    """Test threshold-related edge cases"""
    print("\n" + "=" * 60)
    print("THRESHOLD EDGE CASE TESTS")
    print("=" * 60)
    
    edge_cases = [
        {"threshold": -0.5, "desc": "Negative threshold"},
        {"threshold": 1.5, "desc": "Threshold > 1"},
        {"threshold": "invalid", "desc": "Non-numeric threshold"},
        {"threshold": None, "desc": "None threshold"}
    ]
    
    for case in edge_cases:
        print(f"\nTesting: {case['desc']} (value: {case['threshold']})")
        try:
            detector = BasicPlagiarismDetector(threshold=case['threshold'])
            result = detector.detect("test document", "test document")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

def test_nltk_dependency():
    """Test NLTK dependency issues"""
    print("\n" + "=" * 60)
    print("NLTK DEPENDENCY TESTS")
    print("=" * 60)
    
    # Test what happens if NLTK data is missing
    print("Current NLTK data status:")
    try:
        import nltk
        print("  NLTK available: Yes")
        
        # Try to use NLTK functions
        nltk.data.find('tokenizers/punkt')
        print("  Punkt tokenizer: Available")
        
        nltk.data.find('corpora/stopwords')  
        print("  Stopwords corpus: Available")
        
    except LookupError as e:
        print(f"  NLTK data missing: {e}")
    except Exception as e:
        print(f"  NLTK error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_edge_cases()
    test_threshold_edge_cases()
    test_nltk_dependency()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF MAJOR LIMITATIONS IDENTIFIED:")
    print("=" * 60)
    print("1. No input validation (None, non-string inputs)")
    print("2. No error handling for preprocessing failures")
    print("3. Performance issues with large documents")  
    print("4. Unicode/encoding issues not handled")
    print("5. Empty documents cause division by zero potential")
    print("6. Only lexical similarity - misses semantic plagiarism")
    print("7. Threshold validation missing")
    print("8. NLTK dependency failures not handled")
    print("9. No logging or debugging information")
    print("10. Memory usage not optimized for large datasets")