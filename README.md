# PlagAi - Plagiarism Detection System

A Python-based plagiarism detection tool that analyzes text documents for similarity and potential plagiarism using natural language processing techniques.

## Overview

PlagAi is designed to detect plagiarism between documents by analyzing lexical similarities and word overlap patterns. The system provides configurable similarity thresholds and can process multiple documents simultaneously for comprehensive plagiarism analysis.

## Features

### Current Capabilities
- **Text Preprocessing**: Automatic cleaning, tokenization, and stop word removal
- **Multiple Similarity Metrics**: 
        - Jaccard similarity
        - Overlap coefficient
        - Simple word overlap analysis
        - **TF-IDF cosine similarity** (newly implemented)
        - **N-gram similarity analysis** (2-gram, 3-gram, etc.)
- **Configurable Detection**: Adjustable similarity thresholds (0.0 - 1.0)
- **Batch Processing**: Analyze multiple documents simultaneously
- **Comprehensive Testing**: Edge case handling and limitation analysis
- **Robust Input Validation**: Error handling for various input types and edge cases

### What PlagAi Can Detect
- Direct text copying with minor modifications
- Word-for-word plagiarism
- Documents with high lexical overlap
- Structural similarities in text patterns
- **Sequential phrase similarities using n-grams**
- **Weighted term importance using TF-IDF**

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **NLTK**: Natural language processing and tokenization
- **scikit-learn**: TF-IDF vectorization and machine learning utilities
- **NumPy/Pandas**: Data manipulation and analysis

### Setup
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Basic Detection
```python
from src.detector import BasicPlagiarismDetector

# Initialize detector with custom threshold
detector = BasicPlagiarismDetector(threshold=0.5)

# Compare two documents
doc1 = "Artificial intelligence is transforming the world quickly."
doc2 = "AI is changing the world significantly and rapidly."

result = detector.detect(doc1, doc2)
print(f"Similarity: {result['similarity']:.2f}")
print(f"Plagiarized: {result['is_plagiarized']}")
```

### Multiple Document Analysis
```python
documents = [
                "Machine learning algorithms are powerful tools",
                "ML algorithms are very powerful instruments", 
                "Deep learning is a subset of machine learning"
]

results = detector.analyze_documents(documents)
for result in results:
                print(f"Docs {result['doc1_index']}-{result['doc2_index']}: {result['similarity']:.2f}")
```

### Advanced Similarity Analysis
```python
from src.similarity import ngram_similarity, tfidf_cosine_similarity
from src.preprocessing import preprocess_doc

# N-gram similarity for phrase detection
tokens1 = preprocess_doc("The quick brown fox jumps")
tokens2 = preprocess_doc("Quick brown fox leaps over")

bigram_sim = ngram_similarity(tokens1, tokens2, n=2)
trigram_sim = ngram_similarity(tokens1, tokens2, n=3)

print(f"Bigram similarity: {bigram_sim:.2f}")
print(f"Trigram similarity: {trigram_sim:.2f}")
```

### Custom Preprocessing
```python
from src.preprocessing import preprocess_doc

text = "Hello, world! This is a test document."
tokens = preprocess_doc(text)
print(f"Processed tokens: {tokens}")
```

## System Architecture

```
PlagAi/
├── src/
│   ├── detector.py           # Main plagiarism detection logic
│   ├── preprocessing.py      # Text cleaning and tokenization
│   └── similarity.py         # Similarity calculation algorithms
├── tests/
│   ├── test_detector.py      # Basic functionality tests
│   ├── test_preprocessing.py # Preprocessing pipeline tests
│   ├── test_similarity.py    # Similarity metric tests
│   ├── tf_idf_testing.py     # TF-IDF implementation tests
│   ├── ngrams_testing.py     # N-gram similarity tests
│   └── limitation_testing.py # Edge cases and system limits
└── requirements.txt          # Project dependencies
```

## Current Implementation Status

### ✅ Completed Features
- Basic plagiarism detection with Jaccard similarity
- Text preprocessing pipeline with NLTK
- Input validation and error handling
- Multiple document batch analysis
- TF-IDF calculation and cosine similarity
- N-gram similarity analysis (bigrams, trigrams, etc.)
- Comprehensive edge case testing
- Performance benchmarking for large documents

### 🔧 Recently Added
- **Enhanced Similarity Metrics**: TF-IDF cosine similarity for weighted term analysis
- **N-gram Analysis**: Sequential pattern detection using bigrams and trigrams
- **Robust Error Handling**: Input validation for None, empty, and non-string inputs
- **Performance Testing**: Benchmarks for documents up to 10,000 words
- **Unicode Support**: Handling of special characters and international text

## Current Limitations

### Identified Challenges
- **Semantic Blindness**: Cannot detect paraphrased content with identical meaning
- **Word Order Insensitive**: May miss some structural plagiarism patterns
- **Language Specific**: Currently optimized for English text only
- **Performance**: Processing time increases significantly with very large documents
- **File Format**: Only processes plain text strings (no PDF/DOCX support yet)

### Edge Cases Handled
- Empty and None document inputs
- Non-string input validation
- Unicode and special character processing
- Threshold boundary validation (0.0-1.0)
- NLTK dependency management
- Large document memory optimization

## Testing

Run the comprehensive test suite:

```bash
# Basic functionality tests
python tests/test_detector.py
python tests/test_preprocessing.py
python tests/test_similarity.py

# Advanced feature testing
python tests/tf_idf_testing.py     # TF-IDF implementation verification
python tests/ngrams_testing.py    # N-gram similarity testing
python tests/limitation_testing.py # System boundaries and edge cases
```

### Test Coverage
- Input validation and error handling
- Performance testing with large documents (up to 10,000 words)
- Unicode and special character handling
- Threshold edge cases (negative, >1.0, non-numeric)
- NLTK dependency verification
- Memory usage optimization
- N-gram similarity accuracy
- TF-IDF calculation correctness

## Performance Benchmarks

| Document Size | Processing Time | Memory Usage | Accuracy Notes |
|---------------|----------------|--------------|----------------|
| 100 words     | ~0.01s         | ~1KB         | High accuracy  |
| 1,000 words   | ~0.05s         | ~10KB        | Good performance |
| 5,000 words   | ~0.2s          | ~50KB        | Acceptable for batch |
| 10,000 words  | ~0.8s          | ~200KB       | Edge of practical use |

## Future Enhancements

### High Priority (Next Phase)
- **Semantic Analysis**: Integration of sentence transformers for meaning-based detection
- **Multi-format Support**: PDF, DOCX, and HTML document processing
- **Advanced N-gram Optimization**: Sliding window improvements and skip-grams
- **Citation Detection**: Distinguish between proper attribution and plagiarism

### Medium Priority
- **Web Interface**: User-friendly web application with file upload
- **Database Integration**: Document storage and historical comparison
- **Multi-language Support**: Extend beyond English text processing
- **Real-time Processing**: Stream processing for large document sets

### Long-term Roadmap
1. **Phase 1**: Semantic similarity using transformer models
2. **Phase 2**: Web interface and REST API development
3. **Phase 3**: Multi-format document support and cloud deployment
4. **Phase 4**: Advanced analytics dashboard and reporting

## Contributing

This project is in active development. Current focus areas:
- Semantic similarity algorithms (transformer-based)
- Performance optimization for large datasets
- Additional file format support (PDF, DOCX)
- Advanced preprocessing techniques
- User interface development

### Development Setup
```bash
git clone https://github.com/mal0101/PlagAi.git
cd PlagAi
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

## Technical Implementation

### Similarity Calculation Methods
The system now supports multiple similarity approaches:

1. **Jaccard Similarity**: `|A ∩ B| / |A ∪ B|` - Basic set overlap
2. **Overlap Coefficient**: `|A ∩ B| / min(|A|, |B|)` - Normalized overlap
3. **TF-IDF Cosine Similarity**: Weighted term importance with cosine distance
4. **N-gram Similarity**: Sequential pattern matching (configurable n-size)

### Preprocessing Pipeline
1. Text normalization (lowercase, punctuation removal)
2. Tokenization using NLTK word_tokenize
3. Stop word removal (English corpus)
4. Token filtering and validation
5. Unicode and special character handling

### Error Handling
- Input type validation (string, None, empty checks)
- Threshold boundary enforcement (0.0-1.0)
- NLTK dependency verification
- Memory usage monitoring for large documents
- Graceful degradation for edge cases

## License

This project is developed for educational and research purposes.

## Contact

For questions, suggestions, or contributions, please refer to the project repository.

---

**Note**: PlagAi is currently in active development with focus on expanding similarity detection methods and improving performance. The system is optimized for academic and research use cases, with production deployment requiring additional security and scalability considerations.