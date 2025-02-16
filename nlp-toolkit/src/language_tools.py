from typing import Dict, List, Union, Optional
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import spacy
from textblob import TextBlob
import re
import logging

# Set seed for reproducibility in language detection
DetectorFactory.seed = 0

class LanguageTools:
    """Advanced language processing and translation tools."""
    
    def __init__(self):
        """Initialize language tools with required models."""
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = logging.getLogger(__name__)
        
        # Initialize translation model for specific language pairs
        self.translation_models = {}
        self.tokenizers = {}
        
        # Common language pairs
        self.common_pairs = [
            ('en', 'fr'),  # English to French
            ('en', 'es'),  # English to Spanish
            ('en', 'de')   # English to German
        ]
        
        # Load models for common language pairs
        for src, tgt in self.common_pairs:
            model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
            try:
                self.translation_models[(src, tgt)] = MarianMTModel.from_pretrained(model_name)
                self.tokenizers[(src, tgt)] = MarianTokenizer.from_pretrained(model_name)
            except Exception as e:
                self.logger.warning(f"Could not load model for {src}-{tgt}: {str(e)}")
    
    def detect_language(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detected language and confidence
        """
        try:
            lang = detect(text)
            # Use TextBlob for additional verification
            blob = TextBlob(text)
            blob_lang = blob.detect_language()
            
            confidence = 1.0 if lang == blob_lang else 0.5
            
            return {
                'language': lang,
                'confidence': confidence,
                'verified': lang == blob_lang
            }
        except Exception as e:
            self.logger.error(f"Language detection error: {str(e)}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'verified': False
            }
    
    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        use_local: bool = True
    ) -> Dict[str, str]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (optional)
            use_local: Whether to use local models when available
            
        Returns:
            Dictionary with translation results
        """
        try:
            # Detect source language if not provided
            if not source_lang:
                source_lang = self.detect_language(text)['language']
            
            # Check if we have a local model for this language pair
            lang_pair = (source_lang, target_lang)
            if use_local and lang_pair in self.translation_models:
                translation = self._translate_with_local_model(text, lang_pair)
            else:
                # Fallback to Google Translate
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translation = translator.translate(text)
            
            return {
                'original': text,
                'translated': translation,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'local' if use_local and lang_pair in self.translation_models else 'api'
            }
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return {
                'original': text,
                'translated': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'method': 'failed',
                'error': str(e)
            }
    
    def check_grammar(self, text: str) -> Dict[str, Union[List[Dict], float]]:
        """
        Check grammar and suggest corrections.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with grammar check results
        """
        doc = self.nlp(text)
        blob = TextBlob(text)
        
        # Find potential grammar issues
        issues = []
        
        # Check subject-verb agreement
        for sent in doc.sents:
            subject = None
            verb = None
            
            for token in sent:
                if token.dep_ == "nsubj":
                    subject = token
                elif token.pos_ == "VERB":
                    verb = token
                
                if subject and verb:
                    agreement = self._check_subject_verb_agreement(subject, verb)
                    if not agreement['valid']:
                        issues.append(agreement)
                    subject = None
                    verb = None
        
        # Check for common errors using patterns
        patterns = self._get_grammar_patterns()
        for pattern, error_type in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append({
                    'type': error_type,
                    'text': match.group(),
                    'position': match.span(),
                    'suggestion': self._get_suggestion(match.group(), error_type)
                })
        
        # Calculate overall score
        score = 1.0 - (len(issues) / max(1, len(list(doc.sents))))
        
        return {
            'score': score,
            'issues': issues,
            'corrections': blob.correct(),
            'suggestions': self._generate_suggestions(issues)
        }
    
    def correct_text(
        self,
        text: str,
        fix_spelling: bool = True,
        fix_grammar: bool = True,
        fix_punctuation: bool = True
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Correct text issues.
        
        Args:
            text: Input text
            fix_spelling: Whether to fix spelling
            fix_grammar: Whether to fix grammar
            fix_punctuation: Whether to fix punctuation
            
        Returns:
            Dictionary with correction results
        """
        original = text
        corrections = []
        
        if fix_spelling:
            # Fix spelling using TextBlob
            blob = TextBlob(text)
            corrected = str(blob.correct())
            if corrected != text:
                corrections.append({
                    'type': 'spelling',
                    'original': text,
                    'corrected': corrected
                })
                text = corrected
        
        if fix_grammar:
            # Apply grammar corrections
            grammar_check = self.check_grammar(text)
            for issue in grammar_check['issues']:
                if 'suggestion' in issue and issue['suggestion']:
                    text = text.replace(issue['text'], issue['suggestion'])
                    corrections.append({
                        'type': 'grammar',
                        'original': issue['text'],
                        'corrected': issue['suggestion']
                    })
        
        if fix_punctuation:
            # Fix basic punctuation issues
            text = self._fix_punctuation(text)
        
        return {
            'original': original,
            'corrected': text,
            'corrections': corrections,
            'improvement_score': self._calculate_improvement_score(original, text)
        }
    
    def _translate_with_local_model(self, text: str, lang_pair: tuple) -> str:
        """Translate text using local Marian model."""
        model = self.translation_models[lang_pair]
        tokenizer = self.tokenizers[lang_pair]
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
    
    def _check_subject_verb_agreement(
        self,
        subject: spacy.tokens.Token,
        verb: spacy.tokens.Token
    ) -> Dict[str, Union[bool, str]]:
        """Check if subject and verb agree in number."""
        valid = True
        message = ""
        
        # Simple agreement rules
        if subject.morph.get("Number") and verb.morph.get("Number"):
            if subject.morph.get("Number")[0] != verb.morph.get("Number")[0]:
                valid = False
                message = f"Subject '{subject.text}' and verb '{verb.text}' don't agree in number"
        
        return {
            'valid': valid,
            'subject': subject.text,
            'verb': verb.text,
            'message': message
        }
    
    def _get_grammar_patterns(self) -> Dict[str, str]:
        """Get patterns for common grammar errors."""
        return {
            r'\b(a)\s+[aeiou]': 'article',
            r'\b(me|him|her|them)\s+(is|are|was|were)\b': 'pronoun_verb',
            r'\b(between)\s+([^.]+?)\s+(and)\s+([^.]+?)\b': 'between_and',
            r'\b(less)\s+([^.]+?)\s+(plural)\b': 'less_fewer',
            r'\b(its|it's|your|you're|their|they're|whose|who's)\b': 'homophone'
        }
    
    def _get_suggestion(self, text: str, error_type: str) -> str:
        """Get suggestion for a grammar error."""
        suggestions = {
            'article': lambda t: t.replace('a ', 'an ') if re.match(r'a\s+[aeiou]', t) else t,
            'pronoun_verb': lambda t: re.sub(r'\b(me|him|her|them)\s+', 'I ', t),
            'between_and': lambda t: t,  # Keep as is, just flag
            'less_fewer': lambda t: t.replace('less', 'fewer'),
            'homophone': self._fix_homophone
        }
        
        return suggestions.get(error_type, lambda x: x)(text)
    
    def _fix_homophone(self, text: str) -> str:
        """Fix common homophone errors."""
        fixes = {
            "it's": "its" if "belonging to it" in text.lower() else "it's",
            "its": "it's" if any(w in text.lower() for w in ["is", "has", "would"]) else "its",
            "you're": "your" if "belonging to you" in text.lower() else "you're",
            "your": "you're" if any(w in text.lower() for w in ["are", "would"]) else "your",
            "they're": "their" if "belonging to them" in text.lower() else "they're",
            "their": "they're" if any(w in text.lower() for w in ["are", "would"]) else "their",
            "who's": "whose" if "belonging to whom" in text.lower() else "who's",
            "whose": "who's" if any(w in text.lower() for w in ["is", "has"]) else "whose"
        }
        
        return fixes.get(text.lower(), text)
    
    def _generate_suggestions(self, issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Generate suggestions for each type
        for issue_type, type_issues in issue_types.items():
            if issue_type == 'article':
                suggestions.append("Watch out for article usage before vowels")
            elif issue_type == 'pronoun_verb':
                suggestions.append("Check subject-verb agreement with pronouns")
            elif issue_type == 'homophone':
                suggestions.append("Be careful with commonly confused words")
        
        return suggestions
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix common punctuation issues."""
        # Add space after punctuation if missing
        text = re.sub(r'([.!?,:;])([^\s\d])', r'\1 \2', text)
        
        # Remove space before punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        
        # Ensure single space between sentences
        text = re.sub(r'\s*\.\s*([A-Z])', r'. \1', text)
        
        # Fix quotation marks
        text = re.sub(r'(\w)"(\w)', r'\1" \2', text)
        
        return text
    
    def _calculate_improvement_score(self, original: str, corrected: str) -> float:
        """Calculate improvement score based on corrections made."""
        if original == corrected:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(original, corrected)
        max_length = max(len(original), len(corrected))
        
        # Normalize score (1.0 means no changes needed, 0.0 means completely different)
        return 1.0 - (distance / max_length)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
