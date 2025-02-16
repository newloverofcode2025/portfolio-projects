import spacy
from typing import Dict, List, Union, Optional
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import re
from pathlib import Path
import logging

class DocumentProcessor:
    """Advanced document processing and analysis tool."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the DocumentProcessor with required models.
        
        Args:
            language: Language code (default: "en" for English)
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.language = language
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Union[str, List, Dict]]:
        """
        Process a document file and extract information.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document analysis results
        """
        try:
            # Read document
            text = self._read_document(file_path)
            
            # Process text
            doc = self.nlp(text)
            
            return {
                'content': text,
                'summary': self.summarize(text),
                'topics': self.extract_topics(text),
                'structure': self.analyze_structure(doc),
                'metadata': self.extract_metadata(doc),
                'statistics': self.get_document_statistics(doc)
            }
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise
    
    def summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Generate a comprehensive document summary.
        
        Args:
            text: Input text
            ratio: Summary length ratio
            
        Returns:
            Summarized text
        """
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate sentence importance
        importance_scores = self._calculate_sentence_importance(doc)
        
        # Select top sentences
        n_sentences = max(1, int(len(sentences) * ratio))
        top_indices = np.argsort(importance_scores)[-n_sentences:]
        top_indices = sorted(top_indices)  # Sort to maintain original order
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def extract_topics(
        self,
        text: str,
        num_topics: int = 5,
        words_per_topic: int = 10
    ) -> List[Dict[str, Union[float, List[str]]]]:
        """
        Extract main topics from the document.
        
        Args:
            text: Input text
            num_topics: Number of topics to extract
            words_per_topic: Number of words per topic
            
        Returns:
            List of topics with their words and weights
        """
        # Tokenize and clean text
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.lemma_.strip()
        ]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=15
        )
        
        # Extract topics
        topics = []
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, words_per_topic)
            topics.append({
                'id': topic_id,
                'weight': lda_model.get_topic_terms(topic_id)[0][1],
                'words': [word for word, prob in topic_words]
            })
        
        return sorted(topics, key=lambda x: x['weight'], reverse=True)
    
    def analyze_structure(self, doc) -> Dict[str, Union[List, Dict]]:
        """
        Analyze document structure.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary containing structural analysis
        """
        # Analyze paragraphs
        paragraphs = self._split_into_paragraphs(doc.text)
        
        # Analyze sections and subsections
        sections = self._identify_sections(doc.text)
        
        # Analyze discourse structure
        discourse = self._analyze_discourse(doc)
        
        return {
            'paragraphs': len(paragraphs),
            'sections': sections,
            'discourse': discourse,
            'coherence': self._calculate_coherence(paragraphs)
        }
    
    def extract_metadata(self, doc) -> Dict[str, Union[str, List, Dict]]:
        """
        Extract document metadata.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary containing document metadata
        """
        return {
            'language': self.nlp.meta['lang'],
            'entities': self._extract_unique_entities(doc),
            'keywords': self._extract_keywords(doc),
            'references': self._find_references(doc.text),
            'dates': self._extract_dates(doc)
        }
    
    def get_document_statistics(self, doc) -> Dict[str, Union[int, float, Dict]]:
        """
        Calculate document statistics.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary containing document statistics
        """
        # Basic counts
        word_count = len([token for token in doc if not token.is_punct])
        sent_count = len(list(doc.sents))
        
        # Vocabulary statistics
        vocab = set(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct)
        
        return {
            'word_count': word_count,
            'sentence_count': sent_count,
            'vocabulary_size': len(vocab),
            'avg_sentence_length': word_count / max(1, sent_count),
            'readability': self._calculate_readability(doc),
            'complexity': self._calculate_complexity(doc)
        }
    
    def _read_document(self, file_path: Union[str, Path]) -> str:
        """Read and preprocess document content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _calculate_sentence_importance(self, doc) -> np.ndarray:
        """Calculate importance score for each sentence."""
        sentences = list(doc.sents)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([sent.text for sent in sentences])
        
        # Calculate sentence scores
        scores = np.zeros(len(sentences))
        
        for i, sent in enumerate(sentences):
            # Position score
            position_score = 1.0 / (i + 1)
            
            # Length score
            length_score = len(sent) / len(doc)
            
            # TF-IDF score
            tfidf_score = np.mean(tfidf_matrix[i].toarray())
            
            # Named entity score
            entity_score = len(sent.ents) / max(1, len(sent))
            
            # Combine scores
            scores[i] = (0.3 * position_score +
                        0.2 * length_score +
                        0.3 * tfidf_score +
                        0.2 * entity_score)
        
        return scores
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def _identify_sections(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """Identify document sections and their hierarchy."""
        # Simple section detection based on common patterns
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(\d+\.)+\s+(.+)$',  # Numbered sections
            r'^[A-Z][^.!?]+:',  # Capitalized labels
        ]
        
        sections = []
        for line in text.split('\n'):
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    sections.append({
                        'title': match.group(1) if len(match.groups()) > 0 else line.strip(),
                        'level': line.count('#') if '#' in line else 1
                    })
        
        return sections
    
    def _analyze_discourse(self, doc) -> Dict[str, List[str]]:
        """Analyze discourse markers and structure."""
        discourse_markers = defaultdict(list)
        
        # Common discourse marker categories
        categories = {
            'sequence': ['first', 'then', 'finally', 'next'],
            'contrast': ['however', 'but', 'although', 'nevertheless'],
            'cause_effect': ['therefore', 'thus', 'consequently', 'because'],
            'addition': ['moreover', 'furthermore', 'additionally', 'also'],
            'example': ['for example', 'for instance', 'such as', 'specifically']
        }
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for category, markers in categories.items():
                for marker in markers:
                    if marker in sent_text:
                        discourse_markers[category].append(sent.text)
        
        return dict(discourse_markers)
    
    def _calculate_coherence(self, paragraphs: List[str]) -> float:
        """Calculate text coherence score."""
        if len(paragraphs) <= 1:
            return 1.0
        
        # Vectorize paragraphs
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        
        # Calculate similarity between consecutive paragraphs
        similarities = []
        for i in range(len(paragraphs) - 1):
            similarity = np.dot(
                tfidf_matrix[i].toarray().flatten(),
                tfidf_matrix[i + 1].toarray().flatten()
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def _extract_unique_entities(self, doc) -> Dict[str, List[str]]:
        """Extract unique named entities by type."""
        entities = defaultdict(set)
        for ent in doc.ents:
            entities[ent.label_].add(ent.text)
        return {k: list(v) for k, v in entities.items()}
    
    def _extract_keywords(self, doc, top_n: int = 20) -> List[Dict[str, Union[str, float]]]:
        """Extract keywords using TextRank-like algorithm."""
        # Build word graph
        word_scores = defaultdict(float)
        window_size = 4
        
        words = [token for token in doc if not token.is_stop and not token.is_punct]
        
        for i in range(len(words)):
            window_words = words[max(0, i - window_size):min(len(words), i + window_size)]
            for word in window_words:
                if word != words[i]:
                    word_scores[words[i].text] += 1.0 / max(1, len(window_words))
        
        # Normalize scores
        max_score = max(word_scores.values()) if word_scores else 1.0
        word_scores = {word: score/max_score for word, score in word_scores.items()}
        
        # Sort and return top keywords
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {'word': word, 'score': score}
            for word, score in sorted_words[:top_n]
        ]
    
    def _find_references(self, text: str) -> List[str]:
        """Find potential references or citations."""
        reference_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+,\s*\d{4}\)',  # (Author, 2020)
            r'(?<=[^A-Za-z0-9])\d{4}(?=[^A-Za-z0-9])',  # Years
        ]
        
        references = []
        for pattern in reference_patterns:
            references.extend(re.findall(pattern, text))
        
        return sorted(set(references))
    
    def _extract_dates(self, doc) -> List[Dict[str, str]]:
        """Extract dates mentioned in the document."""
        date_ents = [ent for ent in doc.ents if ent.label_ == 'DATE']
        return [{'text': ent.text, 'context': ent.sent.text} for ent in date_ents]
    
    def _calculate_readability(self, doc) -> Dict[str, float]:
        """Calculate various readability metrics."""
        words = [token for token in doc if not token.is_punct]
        sentences = list(doc.sents)
        
        if not words or not sentences:
            return {'score': 0.0, 'grade_level': 0.0}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = np.mean([len(word.text) for word in words])
        
        # Simple readability score (higher is more complex)
        score = 0.4 * (avg_sentence_length + 100 * (avg_word_length / 5))
        
        return {
            'score': score,
            'grade_level': score / 10  # Approximate grade level
        }
    
    def _calculate_complexity(self, doc) -> Dict[str, float]:
        """Calculate document complexity metrics."""
        # Lexical diversity
        words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        unique_words = set(words)
        
        # Syntactic complexity
        depths = [token.dep_.count('>') for token in doc]  # Dependency tree depth
        
        return {
            'lexical_diversity': len(unique_words) / max(1, len(words)),
            'avg_tree_depth': np.mean(depths) if depths else 0.0,
            'unique_pos_ratio': len(set(token.pos_ for token in doc)) / max(1, len(doc))
        }
