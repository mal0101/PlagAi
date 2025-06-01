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
    - TF-IDF calculation support
- **Configurable Detection**: Adjustable similarity thresholds (0.0 - 1.0)
- **Batch Processing**: Analyze multiple documents simultaneously
- **Comprehensive Testing**: Edge case handling and limitation analysis

### What PlagAi Can Detect
- Direct text copying with minor modifications
- Word-for-word plagiarism
- Documents with high lexical overlap
- Structural similarities in text patterns

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
│   └── limitation_testing.py # Edge cases and system limits
└── requirements.txt          # Project dependencies
```

## Current Limitations

### Identified Challenges
- **Semantic Blindness**: Cannot detect paraphrased content with identical meaning
- **Word Order Insensitive**: May miss structural plagiarism
- **Language Specific**: Currently optimized for English text only
- **Performance**: Not optimized for very large documents (>10,000 words)
- **File Format**: Only processes plain text strings

### Edge Cases Handled
- Empty and None document inputs
- Non-string input validation
- Unicode and special character processing
- Threshold boundary validation
- NLTK dependency management

## Testing

Run the comprehensive test suite:

```bash
# Basic functionality tests
python tests/test_detector.py
python tests/test_preprocessing.py
python tests/test_similarity.py

# Advanced testing
python tests/limitation_testing.py  # Identifies system boundaries
python tests/tf_idf_testing.py     # TF-IDF implementation verification
```

### Test Coverage
- Input validation and error handling
- Performance testing with large documents
- Unicode and special character handling
- Threshold edge cases
- NLTK dependency verification

## Performance Benchmarks

| Document Size | Processing Time | Memory Usage |
|---------------|----------------|--------------|
| 100 words     | ~0.01s         | ~1KB         |
| 1,000 words   | ~0.05s         | ~10KB        |
| 5,000 words   | ~0.2s          | ~50KB        |

## Future Enhancements

### Planned Features
- **Semantic Analysis**: Integration of word embeddings and transformer models
- **Multi-format Support**: PDF, DOCX, and HTML document processing
- **Advanced Metrics**: N-gram analysis and longest common subsequence
- **Web Interface**: User-friendly web application
- **Database Integration**: Document storage and historical comparison
- **Citation Detection**: Proper attribution vs. plagiarism differentiation

### Roadmap
1. **Phase 1**: Semantic similarity using sentence transformers
2. **Phase 2**: Multi-format document support
3. **Phase 3**: Web interface and API development
4. **Phase 4**: Advanced analytics and reporting

## Contributing

This project is in active development. Key areas for contribution:
- Semantic similarity algorithms
- Performance optimization
- Additional file format support
- Advanced preprocessing techniques
- User interface development

## Technical Notes

### Similarity Calculation
The system uses multiple approaches to calculate document similarity:

1. **Jaccard Similarity**: `|A ∩ B| / |A ∪ B|`
2. **Overlap Coefficient**: `|A ∩ B| / min(|A|, |B|)`
3. **TF-IDF Weighting**: Term frequency-inverse document frequency analysis

### Preprocessing Pipeline
1. Text normalization (lowercase, punctuation removal)
2. Tokenization using NLTK
3. Stop word removal
4. Token filtering and cleaning

## License

This project is developed for educational and research purposes.

## Contact

For questions, suggestions, or contributions, please refer to the project repository.

---

**Note**: PlagAi is currently in development and optimized for academic and research use cases. For production deployment, additional security, performance, and scalability considerations should be implemented.