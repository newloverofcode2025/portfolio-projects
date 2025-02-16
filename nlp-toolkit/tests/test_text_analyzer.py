import unittest
from src.text_analyzer import TextAnalyzer
import spacy
import numpy as np

class TestTextAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.analyzer = TextAnalyzer()
        cls.test_text = """
        The quick brown fox jumps over the lazy dog. This is a simple test text
        that contains multiple sentences. AI technology is amazing and transformative,
        but we must be cautious about its implications.
        """
    
    def test_analyze(self):
        """Test the main analyze method."""
        result = self.analyzer.analyze(self.test_text)
        
        # Check if all expected keys are present
        expected_keys = {
            'sentiment', 'entities', 'key_phrases', 'summary',
            'pos_tags', 'readability', 'statistics'
        }
        self.assertEqual(set(result.keys()), expected_keys)
    
    def test_get_sentiment(self):
        """Test sentiment analysis."""
        # Test positive sentiment
        positive_text = "I love this amazing product! It's wonderful."
        pos_sentiment = self.analyzer.get_sentiment(positive_text)
        self.assertEqual(pos_sentiment['label'], 'POSITIVE')
        self.assertGreater(pos_sentiment['score'], 0.5)
        
        # Test negative sentiment
        negative_text = "This is terrible. I hate it completely."
        neg_sentiment = self.analyzer.get_sentiment(negative_text)
        self.assertEqual(neg_sentiment['label'], 'NEGATIVE')
        self.assertGreater(neg_sentiment['score'], 0.5)
    
    def test_get_entities(self):
        """Test named entity recognition."""
        text = "Microsoft and Google are working on AI in New York."
        doc = self.analyzer.nlp(text)
        entities = self.analyzer.get_entities(doc)
        
        # Check if entities are found
        self.assertGreater(len(entities), 0)
        
        # Check entity structure
        entity = entities[0]
        self.assertTrue(all(key in entity for key in ['text', 'label', 'start', 'end']))
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        doc = self.analyzer.nlp(self.test_text)
        phrases = self.analyzer.extract_key_phrases(doc)
        
        # Check if phrases are returned
        self.assertGreater(len(phrases), 0)
        
        # Check phrase structure
        phrase = phrases[0]
        self.assertTrue(all(key in phrase for key in ['phrase', 'score']))
        
        # Check if scores are normalized
        scores = [p['score'] for p in phrases]
        self.assertTrue(all(0 <= score <= 1 for score in scores))
    
    def test_get_summary(self):
        """Test text summarization."""
        summary = self.analyzer.get_summary(self.test_text)
        
        # Check if summary is not empty
        self.assertTrue(len(summary) > 0)
        
        # Check if summary is shorter than original text
        self.assertLess(len(summary), len(self.test_text))
    
    def test_get_pos_tags(self):
        """Test part-of-speech tagging."""
        doc = self.analyzer.nlp(self.test_text)
        pos_tags = self.analyzer.get_pos_tags(doc)
        
        # Check if POS tags are returned
        self.assertGreater(len(pos_tags), 0)
        
        # Check if common POS tags are present
        common_tags = {'NOUN', 'VERB', 'ADJ', 'DET'}
        self.assertTrue(any(tag in pos_tags for tag in common_tags))
    
    def test_analyze_readability(self):
        """Test readability analysis."""
        readability = self.analyzer.analyze_readability(self.test_text)
        
        # Check if all metrics are present
        expected_metrics = {
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'avg_words_per_sentence',
            'avg_syllables_per_word'
        }
        self.assertEqual(set(readability.keys()), expected_metrics)
        
        # Check if metrics are within expected ranges
        self.assertTrue(0 <= readability['flesch_reading_ease'] <= 100)
        self.assertTrue(readability['flesch_kincaid_grade'] >= 0)
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        doc = self.analyzer.nlp(self.test_text)
        stats = self.analyzer.get_text_statistics(doc)
        
        # Check if all statistics are present
        expected_stats = {
            'word_count',
            'unique_words',
            'vocabulary_richness',
            'sentence_count',
            'avg_word_length',
            'top_words'
        }
        self.assertEqual(set(stats.keys()), expected_stats)
        
        # Check if statistics are valid
        self.assertGreater(stats['word_count'], 0)
        self.assertGreater(stats['unique_words'], 0)
        self.assertTrue(0 <= stats['vocabulary_richness'] <= 1)
        self.assertGreater(stats['sentence_count'], 0)
        self.assertTrue(isinstance(stats['top_words'], dict))

if __name__ == '__main__':
    unittest.main()
