import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# First, we start by downloading the NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Clean and standardize the input text by removing punctuation, converting to lowercase,
    and removing whitespace
    """
    # lowercase conversion
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))
    # remove extra whitespace
    text = re.sub(r'\s+',' ',text).strip()
    return text

def tokenize_text(text):
    """Tokenize the text and remove stop words"""
    tokens = word_tokenize(text)
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def preprocess_doc(text):
    """Complete preprocessing pipeline"""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    return tokens