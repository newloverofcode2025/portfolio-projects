import unittest
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_text_analyzer import TestTextAnalyzer
from test_document_processor import TestDocumentProcessor
from test_language_tools import TestLanguageTools

def run_tests():
    """Run all test cases."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTextAnalyzer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocumentProcessor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanguageTools))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())
