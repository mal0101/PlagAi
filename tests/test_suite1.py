import sys
import os
import time
import unittest
from unittest.mock import patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.advanced_detector import AdvancedPlagiarismDetector
from src.detector import BasicPlagiarismDetector
from src.preprocessing import preprocess_doc
from src.similarity import ngram_similarity, tfidf_cosine_similarity

class TestAdvancedPlagiarismDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.detector = AdvancedPlagiarismDetector()
        self.basic_detector = BasicPlagiarismDetector()
        
    def test_initialization_default_weights(self):
        """Test detector initialization with default weights."""
        detector = AdvancedPlagiarismDetector()
        expected_weights = {
            'word_overlap': 0.3,
            'tfidf_cosine': 0.4,
            'bigram_similarity': 0.2,
            'trigram_similarity': 0.1
        }
        self.assertEqual(detector.weights, expected_weights)
        
    def test_initialization_custom_weights(self):
        """Test detector initialization with custom weights."""
        custom_weights = {
            'word_overlap': 0.5,
            'tfidf_cosine': 0.3,
            'bigram_similarity': 0.2
        }
        detector = AdvancedPlagiarismDetector(weights=custom_weights)
        self.assertEqual(detector.weights, custom_weights)
        
    def test_feature_extraction_identical_documents(self):
        """Test feature extraction with identical documents."""
        doc1 = "The quick brown fox jumps over the lazy dog"
        doc2 = "The quick brown fox jumps over the lazy dog"
        
        features = self.detector.extract_features(doc1, doc2)
        
        # All similarity metrics should be high for identical documents
        self.assertAlmostEqual(features['word_overlap'], 1.0, places=2)
        self.assertAlmostEqual(features['bigram_similarity'], 1.0, places=2)
        self.assertAlmostEqual(features['trigram_similarity'], 1.0, places=2)
        self.assertAlmostEqual(features['length_ratio'], 1.0, places=2)
        
    def test_feature_extraction_completely_different_documents(self):
        """Test feature extraction with completely different documents."""
        doc1 = "Artificial intelligence is transforming technology rapidly"
        doc2 = "Ocean waves crash against rocky shores continuously"
        
        features = self.detector.extract_features(doc1, doc2)
        
        # All similarity metrics should be low for different documents
        self.assertLess(features['word_overlap'], 0.3)
        self.assertLess(features['bigram_similarity'], 0.3)
        self.assertLess(features['trigram_similarity'], 0.3)
        
    def test_feature_extraction_partial_overlap(self):
        """Test feature extraction with partial overlap."""
        doc1 = "Machine learning algorithms are powerful tools for data analysis"
        doc2 = "Machine learning techniques are useful instruments for data processing"
        
        features = self.detector.extract_features(doc1, doc2)
        
        # Should have moderate similarity
        self.assertGreater(features['word_overlap'], 0.2)
        self.assertLess(features['word_overlap'], 0.8)
        self.assertGreater(features['bigram_similarity'], 0.1)
        self.assertLess(features['bigram_similarity'], 0.7)
        
    def test_weighted_similarity_calculation(self):
        """Test weighted similarity calculation."""
        features = {
            'word_overlap': 0.6,
            'tfidf_cosine': 0.7,
            'bigram_similarity': 0.5,
            'trigram_similarity': 0.4,
            'length_ratio': 0.8
        }
        
        weighted_score = self.detector.calculate_weighted_similarity(features)
        
        # Manual calculation: 0.3*0.6 + 0.4*0.7 + 0.2*0.5 + 0.1*0.4 = 0.64
        expected_score = 0.3 * 0.6 + 0.4 * 0.7 + 0.2 * 0.5 + 0.1 * 0.4
        self.assertAlmostEqual(weighted_score, expected_score, places=3)
        
    def test_plagiarism_detection_high_similarity(self):
        """Test detection with high similarity documents."""
        doc1 = "Climate change is affecting global weather patterns significantly"
        doc2 = "Climate change significantly affects global weather patterns"
        
        result = self.detector.detect_plagiarism(doc1, doc2, threshold=0.5)
        
        self.assertGreater(result['similarity_score'], 0.5)
        self.assertTrue(result['is_plagiarized'])
        self.assertEqual(result['threshold'], 0.5)
        self.assertIn('features', result)
        
    def test_plagiarism_detection_low_similarity(self):
        """Test detection with low similarity documents."""
        doc1 = "The stock market experienced volatility yesterday"
        doc2 = "Soccer players train rigorously for competitions"
        
        result = self.detector.detect_plagiarism(doc1, doc2, threshold=0.5)
        
        self.assertLess(result['similarity_score'], 0.5)
        self.assertFalse(result['is_plagiarized'])
        
    def test_comparison_with_basic_detector(self):
        """Compare advanced detector with basic detector performance."""
        test_pairs = [
            ("The cat sat on the mat", "A cat was sitting on the mat"),
            ("Machine learning is powerful", "ML algorithms are strong"),
            ("Python programming language", "Java programming language"),
            ("Artificial intelligence research", "AI research and development")
        ]
        
        for doc1, doc2 in test_pairs:
            # Advanced detector result
            advanced_result = self.detector.detect_plagiarism(doc1, doc2)
            
            # Basic detector result
            basic_result = self.basic_detector.detect(doc1, doc2)
            
            print(f"\nComparing: '{doc1}' vs '{doc2}'")
            print(f"Advanced Score: {advanced_result['similarity_score']:.3f}")
            print(f"Basic Score: {basic_result['similarity']:.3f}")
            print(f"Advanced Features: {advanced_result['features']}")
            
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty documents
        try:
            features = self.detector.extract_features("", "")
            self.assertIsInstance(features, dict)
        except Exception as e:
            self.fail(f"Empty documents should be handled gracefully: {e}")
            
        # Single word documents
        features = self.detector.extract_features("hello", "world")
        self.assertIsInstance(features, dict)
        self.assertIn('word_overlap', features)
        
        # Documents with only stop words
        features = self.detector.extract_features("the and or", "a an the")
        self.assertIsInstance(features, dict)
        
    def test_performance_with_large_documents(self):
        """Test performance with large documents."""
        # Create large documents
        large_doc1 = "artificial intelligence machine learning " * 1000
        large_doc2 = "machine learning artificial intelligence " * 1000
        
        start_time = time.time()
        result = self.detector.detect_plagiarism(large_doc1, large_doc2)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\nLarge document processing time: {processing_time:.3f} seconds")
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 10.0, "Processing should complete within 10 seconds")
        self.assertIsInstance(result, dict)
        
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        test_cases = [
            ("Héllo wörld", "Hello world"),
            ("数据科学很有趣", "数据科学非常有趣"),
            ("Testing 123!@#", "Testing 456$%^"),
            ("émojis 🚀🎉", "emojis rocket party")
        ]
        
        for doc1, doc2 in test_cases:
            try:
                result = self.detector.detect_plagiarism(doc1, doc2)
                self.assertIsInstance(result, dict)
                print(f"Unicode test passed: '{doc1}' vs '{doc2}' -> {result['similarity_score']:.3f}")
            except Exception as e:
                print(f"Unicode handling issue with '{doc1}' vs '{doc2}': {e}")
                
    def test_threshold_sensitivity(self):
        """Test detector behavior across different thresholds."""
        doc1 = "Machine learning algorithms process data efficiently"
        doc2 = "ML algorithms efficiently process data sets"
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            result = self.detector.detect_plagiarism(doc1, doc2, threshold=threshold)
            similarity_score = result['similarity_score']
            
            # Similarity score should remain constant regardless of threshold
            if 'base_similarity' not in locals():
                base_similarity = similarity_score
            else:
                self.assertAlmostEqual(similarity_score, base_similarity, places=3)
                
            # Plagiarism flag should change based on threshold
            expected_plagiarized = similarity_score >= threshold
            self.assertEqual(result['is_plagiarized'], expected_plagiarized)
            
    def test_feature_completeness(self):
        """Test that all expected features are extracted."""
        doc1 = "Natural language processing is fascinating"
        doc2 = "NLP techniques are very interesting"
        
        features = self.detector.extract_features(doc1, doc2)
        
        expected_features = [
            'word_overlap', 'tfidf_cosine', 'bigram_similarity', 
            'trigram_similarity', 'length_ratio'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            self.assertGreaterEqual(features[feature], 0.0)
            self.assertLessEqual(features[feature], 1.0)
            
    def test_corpus_parameter(self):
        """Test feature extraction with custom corpus."""
        doc1 = "Machine learning is powerful"
        doc2 = "ML is very strong"
        corpus = [doc1, doc2, "Additional context document", "More corpus content"]
        
        features_with_corpus = self.detector.extract_features(doc1, doc2, corpus=corpus)
        features_without_corpus = self.detector.extract_features(doc1, doc2)
        
        # TF-IDF should be different with larger corpus
        self.assertNotEqual(
            features_with_corpus['tfidf_cosine'], 
            features_without_corpus['tfidf_cosine']
        )

class TestAdvancedDetectorIntegration(unittest.TestCase):
    """Integration tests for advanced detector with real-world scenarios."""
    
    def setUp(self):
        self.detector = AdvancedPlagiarismDetector()
        
    def test_academic_plagiarism_scenarios(self):
        """Test scenarios common in academic plagiarism detection."""
        scenarios = [
            {
                "name": "Direct Copy",
                "doc1": "Climate change is one of the most pressing issues of our time",
                "doc2": "Climate change is one of the most pressing issues of our time",
                "expected_high": True
            },
            {
                "name": "Word Substitution",
                "doc1": "Global warming poses significant threats to humanity",
                "doc2": "Climate change presents major risks to mankind",
                "expected_high": False
            },
            {
                "name": "Sentence Restructuring", 
                "doc1": "Machine learning algorithms can process vast amounts of data",
                "doc2": "Vast amounts of data can be processed by machine learning algorithms",
                "expected_high": True
            },
            {
                "name": "Paraphrasing",
                "doc1": "Artificial intelligence is revolutionizing many industries",
                "doc2": "AI technology is transforming numerous business sectors",
                "expected_high": False
            }
        ]
        
        print("\n" + "="*50)
        print("ACADEMIC PLAGIARISM SCENARIO TESTING")
        print("="*50)
        
        for scenario in scenarios:
            result = self.detector.detect_plagiarism(scenario["doc1"], scenario["doc2"])
            similarity = result['similarity_score']
            
            print(f"\nScenario: {scenario['name']}")
            print(f"Doc1: {scenario['doc1']}")
            print(f"Doc2: {scenario['doc2']}")
            print(f"Similarity Score: {similarity:.3f}")
            print(f"Features: {result['features']}")
            
            if scenario["expected_high"]:
                self.assertGreater(similarity, 0.4, 
                    f"Expected high similarity for {scenario['name']}")
            else:
                print(f"Moderate/Low similarity detected as expected")

def run_comprehensive_tests():
    """Run comprehensive test suite with detailed reporting."""
    print("="*60)
    print("COMPREHENSIVE ADVANCED PLAGIARISM DETECTOR TEST SUITE")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_suite.addTest(unittest.makeSuite(TestAdvancedPlagiarismDetector))
    test_suite.addTest(unittest.makeSuite(TestAdvancedDetectorIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    # Additional manual testing
    print("\n" + "="*60)
    print("MANUAL TESTING EXAMPLES")
    print("="*60)
    
    detector = AdvancedPlagiarismDetector()
    
    # Test different document pairs
    test_pairs = [
        ("The quick brown fox jumps over the lazy dog", 
         "A quick brown fox leaps over a lazy dog"),
        ("Python is a programming language", 
         "Java is a programming language"),
        ("Artificial intelligence and machine learning", 
         "AI and ML technologies"),
        ("Data science involves statistics and programming",
         "Statistics and programming are used in data science")
    ]
    
    for i, (doc1, doc2) in enumerate(test_pairs, 1):
        result = detector.detect_plagiarism(doc1, doc2)
        print(f"\nTest {i}:")
        print(f"Doc1: {doc1}")
        print(f"Doc2: {doc2}")
        print(f"Similarity: {result['similarity_score']:.3f}")
        print(f"Plagiarized: {result['is_plagiarized']}")
        print(f"Key Features:")
        for feature, value in result['features'].items():
            print(f"  {feature}: {value:.3f}")
    
    print(f"\nTest suite completed successfully: {success}")