import unittest
import tempfile
import os
from src.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.processor = DocumentProcessor()
        
        # Create a temporary test document
        cls.test_content = """
        # Machine Learning Overview
        
        Machine learning is a fascinating field of artificial intelligence.
        It enables computers to learn from data and improve over time.
        
        ## Types of Learning
        
        1. Supervised Learning
        2. Unsupervised Learning
        3. Reinforcement Learning
        
        ## Applications
        
        Machine learning has many applications in:
        - Healthcare
        - Finance
        - Technology
        
        The future of ML looks promising.
        """
        
        # Create temporary file
        cls.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        cls.temp_file.write(cls.test_content)
        cls.temp_file.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        os.unlink(cls.temp_file.name)
    
    def test_process_document(self):
        """Test the main document processing method."""
        result = self.processor.process_document(self.temp_file.name)
        
        # Check if all expected components are present
        expected_keys = {
            'content',
            'summary',
            'topics',
            'structure',
            'metadata',
            'statistics'
        }
        self.assertEqual(set(result.keys()), expected_keys)
    
    def test_summarize(self):
        """Test document summarization."""
        summary = self.processor.summarize(self.test_content)
        
        # Check if summary is not empty
        self.assertTrue(len(summary) > 0)
        
        # Check if summary is shorter than original
        self.assertLess(len(summary), len(self.test_content))
        
        # Test with different ratio
        short_summary = self.processor.summarize(self.test_content, ratio=0.1)
        long_summary = self.processor.summarize(self.test_content, ratio=0.5)
        self.assertLess(len(short_summary), len(long_summary))
    
    def test_extract_topics(self):
        """Test topic extraction."""
        topics = self.processor.extract_topics(self.test_content)
        
        # Check if topics are returned
        self.assertTrue(len(topics) > 0)
        
        # Check topic structure
        topic = topics[0]
        self.assertTrue(all(key in topic for key in ['id', 'weight', 'words']))
        
        # Check if weights are normalized
        weights = [t['weight'] for t in topics]
        self.assertTrue(all(0 <= w <= 1 for w in weights))
    
    def test_analyze_structure(self):
        """Test document structure analysis."""
        doc = self.processor.nlp(self.test_content)
        structure = self.processor.analyze_structure(doc)
        
        # Check if all structure components are present
        expected_components = {
            'paragraphs',
            'sections',
            'discourse',
            'coherence'
        }
        self.assertEqual(set(structure.keys()), expected_components)
        
        # Check if coherence score is normalized
        self.assertTrue(0 <= structure['coherence'] <= 1)
        
        # Check if sections are detected
        self.assertTrue(len(structure['sections']) > 0)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        doc = self.processor.nlp(self.test_content)
        metadata = self.processor.extract_metadata(doc)
        
        # Check if all metadata components are present
        expected_components = {
            'language',
            'entities',
            'keywords',
            'references',
            'dates'
        }
        self.assertEqual(set(metadata.keys()), expected_components)
        
        # Check language detection
        self.assertEqual(metadata['language'], 'en')
        
        # Check if keywords are extracted
        self.assertTrue(len(metadata['keywords']) > 0)
    
    def test_get_document_statistics(self):
        """Test document statistics calculation."""
        doc = self.processor.nlp(self.test_content)
        stats = self.processor.get_document_statistics(doc)
        
        # Check if all statistics are present
        expected_stats = {
            'word_count',
            'sentence_count',
            'vocabulary_size',
            'avg_sentence_length',
            'readability',
            'complexity'
        }
        self.assertEqual(set(stats.keys()), expected_stats)
        
        # Check if statistics are valid
        self.assertGreater(stats['word_count'], 0)
        self.assertGreater(stats['sentence_count'], 0)
        self.assertGreater(stats['vocabulary_size'], 0)
        self.assertTrue(isinstance(stats['readability'], dict))
        self.assertTrue(isinstance(stats['complexity'], dict))
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        with self.assertRaises(Exception):
            self.processor.process_document('nonexistent_file.txt')
        
        # Test with empty text
        empty_summary = self.processor.summarize("")
        self.assertEqual(empty_summary, "")
        
        # Test with very short text
        short_text = "Hello world."
        short_topics = self.processor.extract_topics(short_text)
        self.assertTrue(isinstance(short_topics, list))

if __name__ == '__main__':
    unittest.main()
