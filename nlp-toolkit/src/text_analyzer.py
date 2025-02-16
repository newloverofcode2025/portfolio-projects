import spacy
import nltk
from textblob import TextBlob
from typing import Dict, List, Union
import numpy as np
from collections import Counter

class TextAnalyzer:
    """A comprehensive text analysis tool using basic NLP features."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the TextAnalyzer with required models.
        
        Args:
            language: Language code (default: "en" for English)
        """
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def analyze(self, text: str) -> Dict[str, Union[str, List, Dict]]:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Basic analysis using spaCy
        doc = self.nlp(text)
        
        # Get sentiment using TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Get key phrases (noun chunks)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Get POS distribution
        pos_dist = Counter([token.pos_ for token in doc])
        
        return {
            'sentiment': sentiment,
            'entities': entities,
            'key_phrases': key_phrases,
            'pos_distribution': dict(pos_dist),
            'word_count': len([token for token in doc if not token.is_punct]),
            'sentence_count': len(list(doc.sents))
        }
    
    def get_pos_tags(self, doc) -> Dict[str, List[Dict[str, str]]]:
        """
        Get part-of-speech tags for the text.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary of POS tags and their words
        """
        pos_dict = {}
        for token in doc:
            if token.pos_ not in pos_dict:
                pos_dict[token.pos_] = []
            pos_dict[token.pos_].append({
                'text': token.text,
                'lemma': token.lemma_,
                'position': token.i
            })
        return pos_dict
    
    def analyze_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability scores
        """
        doc = self.nlp(text)
        
        # Count sentences, words, and syllables
        sentences = len(list(doc.sents))
        words = len([token for token in doc if not token.is_punct])
        syllables = sum(self._count_syllables(word.text) for word in doc if not word.is_punct)
        
        # Calculate metrics
        if words == 0 or sentences == 0:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'avg_words_per_sentence': 0.0,
                'avg_syllables_per_word': 0.0
            }
        
        avg_words_per_sentence = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Flesch Reading Ease
        flesch = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch)),
            'flesch_kincaid_grade': max(0, flesch_kincaid),
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def get_text_statistics(self, doc) -> Dict[str, Union[int, float, Dict]]:
        """
        Get statistical information about the text.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary of text statistics
        """
        # Word frequency
        word_freq = Counter([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])
        
        # Vocabulary richness
        unique_words = len(set(token.text.lower() for token in doc if not token.is_punct))
        total_words = len([token for token in doc if not token.is_punct])
        
        return {
            'word_count': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': unique_words / max(1, total_words),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(token.text) for token in doc if not token.is_punct]),
            'top_words': dict(word_freq.most_common(10))
        }
    
    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word.
        
        Args:
            word: Input word
            
        Returns:
            Number of syllables
        """
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        
        # Handle special cases
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        prev_char = ''
        for char in word:
            if char in vowels and prev_char not in vowels:
                count += 1
            prev_char = char
        
        return max(1, count)  # Every word has at least one syllable
