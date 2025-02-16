import unittest
from src.language_tools import LanguageTools

class TestLanguageTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.lang_tools = LanguageTools()
    
    def test_detect_language(self):
        """Test language detection."""
        # Test English text
        english_text = "This is a sample English text."
        eng_result = self.lang_tools.detect_language(english_text)
        self.assertEqual(eng_result['language'], 'en')
        self.assertTrue(eng_result['confidence'] > 0.5)
        
        # Test Spanish text
        spanish_text = "Este es un texto de ejemplo en español."
        spa_result = self.lang_tools.detect_language(spanish_text)
        self.assertEqual(spa_result['language'], 'es')
        self.assertTrue(spa_result['confidence'] > 0.5)
        
        # Test French text
        french_text = "C'est un exemple de texte en français."
        fra_result = self.lang_tools.detect_language(french_text)
        self.assertEqual(fra_result['language'], 'fr')
        self.assertTrue(fra_result['confidence'] > 0.5)
    
    def test_translate(self):
        """Test translation functionality."""
        text = "Hello, how are you?"
        
        # Test translation to Spanish
        es_translation = self.lang_tools.translate(text, target_lang='es')
        self.assertTrue(all(key in es_translation for key in [
            'original', 'translated', 'source_lang', 'target_lang', 'method'
        ]))
        self.assertEqual(es_translation['source_lang'], 'en')
        self.assertEqual(es_translation['target_lang'], 'es')
        
        # Test translation with specified source language
        fr_translation = self.lang_tools.translate(
            text,
            target_lang='fr',
            source_lang='en'
        )
        self.assertEqual(fr_translation['source_lang'], 'en')
        self.assertEqual(fr_translation['target_lang'], 'fr')
        
        # Test translation with local model
        local_translation = self.lang_tools.translate(
            text,
            target_lang='de',
            use_local=True
        )
        self.assertIn(local_translation['method'], ['local', 'api'])
    
    def test_check_grammar(self):
        """Test grammar checking functionality."""
        # Test correct grammar
        correct_text = "This is a correctly written sentence."
        correct_result = self.lang_tools.check_grammar(correct_text)
        self.assertTrue(correct_result['score'] > 0.8)
        
        # Test incorrect grammar
        incorrect_text = "This sentence have bad grammar."
        incorrect_result = self.lang_tools.check_grammar(incorrect_text)
        self.assertTrue(incorrect_result['score'] < 0.8)
        self.assertTrue(len(incorrect_result['issues']) > 0)
        
        # Check issue structure
        if incorrect_result['issues']:
            issue = incorrect_result['issues'][0]
            self.assertTrue(all(key in issue for key in ['type', 'text', 'position']))
    
    def test_correct_text(self):
        """Test text correction functionality."""
        # Test spelling correction
        misspelled_text = "This is a mispeled sentense."
        spell_result = self.lang_tools.correct_text(
            misspelled_text,
            fix_spelling=True,
            fix_grammar=False,
            fix_punctuation=False
        )
        self.assertNotEqual(spell_result['original'], spell_result['corrected'])
        
        # Test grammar correction
        grammar_text = "They is going to the store."
        grammar_result = self.lang_tools.correct_text(
            grammar_text,
            fix_spelling=False,
            fix_grammar=True,
            fix_punctuation=False
        )
        self.assertNotEqual(grammar_result['original'], grammar_result['corrected'])
        
        # Test punctuation correction
        punct_text = "This sentence needs punctuation"
        punct_result = self.lang_tools.correct_text(
            punct_text,
            fix_spelling=False,
            fix_grammar=False,
            fix_punctuation=True
        )
        self.assertNotEqual(punct_result['original'], punct_result['corrected'])
        
        # Test all corrections
        text_with_errors = "This sentense have bad grammer and punctuation"
        full_result = self.lang_tools.correct_text(
            text_with_errors,
            fix_spelling=True,
            fix_grammar=True,
            fix_punctuation=True
        )
        self.assertNotEqual(full_result['original'], full_result['corrected'])
        self.assertTrue(len(full_result['corrections']) > 0)
        self.assertTrue(0 <= full_result['improvement_score'] <= 1)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test empty text
        empty_result = self.lang_tools.detect_language("")
        self.assertEqual(empty_result['language'], 'unknown')
        
        # Test invalid language code
        with self.assertRaises(Exception):
            self.lang_tools.translate("Hello", target_lang='invalid')
        
        # Test very long text
        long_text = "word " * 1000
        long_result = self.lang_tools.check_grammar(long_text)
        self.assertTrue(isinstance(long_result['score'], float))
    
    def test_helper_methods(self):
        """Test internal helper methods."""
        # Test subject-verb agreement checking
        doc = self.lang_tools.nlp("They is running.")
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token
            elif token.pos_ == "VERB":
                verb = token
                break
        
        agreement = self.lang_tools._check_subject_verb_agreement(subject, verb)
        self.assertFalse(agreement['valid'])
        
        # Test homophone correction
        homophone_text = "its going to rain"
        corrected = self.lang_tools._fix_homophone("its")
        self.assertEqual(corrected, "it's")
        
        # Test punctuation fixing
        text_without_spaces = "Hello,world!How are you?"
        fixed = self.lang_tools._fix_punctuation(text_without_spaces)
        self.assertIn(" ", fixed)

if __name__ == '__main__':
    unittest.main()
