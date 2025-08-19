#!/usr/bin/env python3
"""
Enhanced Intelligent Web Scraper with Dynamic Content Understanding
==================================================================

This enhanced version focuses on identifying meaningful elements like cards, posts,
buttons, and interactive components with dynamic semantic role assignment based on
content, structure, and attributes.

Features:
- Dynamic semantic role detection based on element structure and content
- Improved identification of interactive elements (cards, buttons, forms)
- Better handling of modern web components
- More accurate topic extraction and entity recognition
- Context-aware content analysis

Author: Manus AI
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import logging
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
import os
import sys
import argparse

# Try to import AI/ML libraries with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Some NLP features will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Using NLTK for NLP tasks.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. ML features will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available. Some calculations will be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except:
            pass

# Load spaCy model if available
nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        try:
            # Try to download the model
            os.system('python3 -m spacy download en_core_web_lg')
            nlp = spacy.load('en_core_web_lg')
        except:
            print("spaCy English model not available. Using NLTK for NLP tasks.")
            nlp = None

@dataclass
class Entity:
    """Represents an extracted entity with its properties."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 0.0
    linked_id: Optional[str] = None

@dataclass
class Relation:
    """Represents a relationship between entities."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0

@dataclass
class ContentBlock:
    """Represents a semantically understood content block."""
    block_id: str
    semantic_role: str
    text_content: str
    html_snippet: Optional[str] = None
    xpath: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    entities: List[Entity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    summary: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    nested_blocks: List["ContentBlock"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactive_elements: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class PageAnalysis:
    """Complete analysis of a web page."""
    url: str
    title: Optional[str] = None
    meta_description: Optional[str] = None
    language: Optional[str] = None
    main_content_blocks: List[ContentBlock] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    overall_sentiment: Dict[str, Any] = field(default_factory=dict)
    page_topics: List[str] = field(default_factory=list)
    content_categories: List[str] = field(default_factory=list)
    processing_time_ms: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentContext:
    """Represents the context of a content block."""
    parent_role: Optional[str] = None
    sibling_roles: List[str] = field(default_factory=list)
    position_in_page: float = 0.0  # 0.0 = top, 1.0 = bottom
    visual_prominence: float = 0.0  # Based on HTML structure
    text_density: float = 0.0
    link_density: float = 0.0
    interactive_score: float = 0.0  # Based on interactive elements

class AIContentAnalyzer:
    """AI-powered content analysis using various ML techniques."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') if SKLEARN_AVAILABLE else None
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities using available NLP libraries."""
        entities = []
        
        if nlp:  # Use spaCy if available
            doc = nlp(text)
            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.8  # spaCy doesn't provide confidence scores directly
                ))
        elif NLTK_AVAILABLE:  # Fallback to NLTK
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_chunk = []
                current_label = None
                char_pos = 0
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):  # Named entity
                        current_label = chunk.label()
                        current_chunk = [token for token, pos in chunk]
                        entity_text = ' '.join(current_chunk)
                        start_char = text.find(entity_text, char_pos)
                        if start_char != -1:
                            entities.append(Entity(
                                text=entity_text,
                                label=current_label,
                                start_char=start_char,
                                end_char=start_char + len(entity_text),
                                confidence=0.6
                            ))
                            char_pos = start_char + len(entity_text)
            except Exception as e:
                logger.warning(f"NLTK entity extraction failed: {e}")
        
        # Add custom entity patterns
        entities.extend(self._extract_custom_entities(text))
        return entities
    
    def _extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract custom entities using regex patterns."""
        entities = []
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'URL': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/._-])*(?:\?[\w&=%.]*)?(?:#[\w.]*)?)?',
            'PRICE': r'\$\d+(?:\.\d{2})?|\d+\s?(?:USD|EUR|GBP|dollars?|euros?|pounds?)\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            'PERCENTAGE': r'\b\d+(?:\.\d+)?%\b'
        }
        
        for label, pattern in patterns.items():
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        text=match.group(),
                        label=label,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.9
                    ))
            except re.error as e:
                logger.warning(f"Regex error for pattern {label}: {e}")
        
        return entities
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text."""
        if not text.strip():
            return []
        
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
            try:
                # Use TF-IDF for keyword extraction
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top keywords
                keyword_scores = list(zip(feature_names, tfidf_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                return [keyword for keyword, score in keyword_scores[:max_keywords] if score > 0]
            except Exception as e:
                logger.warning(f"TF-IDF keyword extraction failed: {e}")
        
        # Fallback to simple frequency-based extraction
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words and len(token) > 3]
                freq_dist = Counter(tokens)
                return [word for word, freq in freq_dist.most_common(max_keywords)]
            except Exception as e:
                logger.warning(f"NLTK keyword extraction failed: {e}")
        
        # Basic fallback
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return list(Counter(words).keys())[:max_keywords]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using available methods."""
        if not text.strip():
            return {"overall": "neutral", "confidence": 0.0}
        
        # Simple rule-based sentiment analysis
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'awesome'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing'}
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.8, positive_count / max(len(words), 1))
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.8, negative_count / max(len(words), 1))
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "overall": sentiment,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def extract_topics(self, text: str, num_topics: int = 3) -> List[str]:
        """Extract main topics from text."""
        if not text.strip():
            return []
        
        # Simple topic extraction based on noun phrases and keywords
        topics = []
        
        if nlp:
            try:
                doc = nlp(text)
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
                topic_candidates = Counter(noun_phrases).most_common(num_topics)
                topics = [topic for topic, count in topic_candidates if count > 1]
            except Exception as e:
                logger.warning(f"spaCy topic extraction failed: {e}")
        
        if not topics and NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                pos_tags = pos_tag(tokens)
                nouns = [word for word, pos in pos_tags if pos.startswith('NN') and len(word) > 3]
                topics = [word for word, count in Counter(nouns).most_common(num_topics)]
            except Exception as e:
                logger.warning(f"NLTK topic extraction failed: {e}")
        
        # Fallback to keywords
        if not topics:
            keywords = self.extract_keywords(text, num_topics)
            topics = keywords[:num_topics]
        
        return topics
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the text."""
        if not text.strip():
            return ""
        
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                if len(sentences) <= max_sentences:
                    return text
                
                # Simple extractive summarization based on sentence importance
                sentence_scores = {}
                keywords = self.extract_keywords(text, 20)
                
                for sentence in sentences:
                    score = 0
                    words = word_tokenize(sentence.lower())
                    for word in words:
                        if word in keywords:
                            score += 1
                    sentence_scores[sentence] = score / len(words) if words else 0
                
                # Select top sentences
                top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
                summary_sentences = [sent for sent, score in top_sentences]
                
                # Maintain original order
                summary = []
                for sentence in sentences:
                    if sentence in summary_sentences:
                        summary.append(sentence)
                
                return ' '.join(summary)
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
        
        # Fallback: return first few sentences
        sentences = re.split(r'[.!?]+', text)
        return '. '.join(sentences[:max_sentences]).strip() + '.'

class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer with dynamic content understanding."""
    
    def __init__(self):
        self.ai_analyzer = AIContentAnalyzer()
        
        # Enhanced semantic patterns with more dynamic detection
        self.semantic_patterns = {
            "project_card": {
                "class_indicators": ["project", "card", "item", "listing", "post"],
                "content_indicators": ["مشروع", "project", "تطوير", "برمجة", "تصميم"],
                "structure_indicators": ["clickable", "has_link", "has_price", "has_title"],
                "score": 0.95
            },
            "job_listing": {
                "class_indicators": ["job", "position", "vacancy", "career"],
                "content_indicators": ["وظيفة", "job", "career", "position", "hiring"],
                "structure_indicators": ["apply", "salary", "requirements"],
                "score": 0.9
            },
            "user_profile": {
                "class_indicators": ["profile", "user", "member", "author"],
                "content_indicators": ["profile", "member", "user", "freelancer"],
                "structure_indicators": ["avatar", "rating", "reviews"],
                "score": 0.85
            },
            "action_button": {
                "class_indicators": ["btn", "button", "action", "submit"],
                "content_indicators": ["submit", "send", "apply", "buy", "order", "أضف", "إرسال"],
                "structure_indicators": ["clickable", "form_element"],
                "score": 0.8
            },
            "price_display": {
                "class_indicators": ["price", "cost", "amount", "fee"],
                "content_indicators": ["$", "€", "£", "ريال", "دولار", "price", "cost"],
                "structure_indicators": ["currency", "numeric"],
                "score": 0.9
            },
            "navigation_item": {
                "class_indicators": ["nav", "menu", "breadcrumb", "tab"],
                "content_indicators": ["home", "about", "contact", "services", "الرئيسية"],
                "structure_indicators": ["link", "menu_item"],
                "score": 0.7
            },
            "content_card": {
                "class_indicators": ["card", "box", "panel", "widget"],
                "content_indicators": ["title", "description", "content"],
                "structure_indicators": ["bordered", "elevated", "clickable"],
                "score": 0.75
            },
            "form_field": {
                "class_indicators": ["input", "field", "form", "control"],
                "content_indicators": ["name", "email", "message", "search"],
                "structure_indicators": ["input_element", "form_control"],
                "score": 0.7
            },
            "rating_display": {
                "class_indicators": ["rating", "stars", "score", "review"],
                "content_indicators": ["stars", "rating", "score", "تقييم"],
                "structure_indicators": ["numeric", "visual_rating"],
                "score": 0.8
            },
            "date_time": {
                "class_indicators": ["date", "time", "timestamp", "ago"],
                "content_indicators": ["منذ", "ago", "yesterday", "today", "hours", "minutes"],
                "structure_indicators": ["temporal"],
                "score": 0.6
            }
        }

    def analyze_semantic_role(self, text: str, html_element, context: ContentContext) -> Tuple[str, float]:
        """Enhanced semantic role analysis with dynamic detection."""
        best_role = "generic_content"
        best_score = 0.1
        
        if not html_element:
            return best_role, best_score
        
        # Get element attributes
        element_classes = ' '.join(html_element.get('class', [])).lower()
        element_id = html_element.get('id', '').lower()
        element_tag = html_element.name.lower()
        
        # Analyze each semantic pattern
        for role, pattern in self.semantic_patterns.items():
            score = self._calculate_dynamic_score(
                text, element_classes, element_id, element_tag, 
                html_element, pattern, context
            )
            
            if score > best_score:
                best_score = score
                best_role = role
        
        # Apply HTML5 semantic element detection
        html5_score, html5_role = self._analyze_html5_semantics(html_element, text)
        if html5_score > best_score:
            best_score = html5_score
            best_role = html5_role
        
        # Apply interactive element detection
        interactive_score, interactive_role = self._analyze_interactive_elements(html_element, text)
        if interactive_score > best_score:
            best_score = interactive_score
            best_role = interactive_role
        
        return best_role, min(1.0, best_score)
    
    def _calculate_dynamic_score(self, text: str, classes: str, element_id: str, 
                                tag: str, element, pattern: Dict, context: ContentContext) -> float:
        """Calculate dynamic score based on multiple factors."""
        score = 0.0
        
        # Class-based scoring
        class_matches = sum(1 for indicator in pattern.get("class_indicators", []) 
                           if indicator in classes or indicator in element_id)
        if class_matches > 0:
            score += (class_matches / len(pattern.get("class_indicators", [1]))) * 0.4
        
        # Content-based scoring
        text_lower = text.lower()
        content_matches = sum(1 for indicator in pattern.get("content_indicators", []) 
                             if indicator in text_lower)
        if content_matches > 0:
            score += (content_matches / len(pattern.get("content_indicators", [1]))) * 0.3
        
        # Structure-based scoring
        structure_score = self._analyze_structure_indicators(element, pattern.get("structure_indicators", []))
        score += structure_score * 0.3
        
        return score * pattern.get("score", 1.0)
    
    def _analyze_structure_indicators(self, element, indicators: List[str]) -> float:
        """Analyze structural indicators of an element."""
        structure_score = 0.0
        total_indicators = len(indicators) if indicators else 1
        
        for indicator in indicators:
            if indicator == "clickable":
                if element.find('a') or element.name == 'a' or element.get('onclick'):
                    structure_score += 1
            elif indicator == "has_link":
                if element.find('a') or element.name == 'a':
                    structure_score += 1
            elif indicator == "has_price":
                if re.search(r'[\$€£]\d+|\d+\s*(?:USD|EUR|GBP|ريال)', element.get_text()):
                    structure_score += 1
            elif indicator == "has_title":
                if element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) or element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    structure_score += 1
            elif indicator == "form_element":
                if element.name in ['input', 'textarea', 'select', 'button'] or element.find(['input', 'textarea', 'select', 'button']):
                    structure_score += 1
            elif indicator == "numeric":
                if re.search(r'\d+', element.get_text()):
                    structure_score += 1
            elif indicator == "currency":
                if re.search(r'[\$€£]|\b(?:USD|EUR|GBP|ريال|دولار)\b', element.get_text()):
                    structure_score += 1
            elif indicator == "temporal":
                if re.search(r'\b(?:منذ|ago|yesterday|today|hours?|minutes?|days?)\b', element.get_text(), re.IGNORECASE):
                    structure_score += 1
        
        return structure_score / total_indicators
    
    def _analyze_html5_semantics(self, element, text: str) -> Tuple[float, str]:
        """Analyze HTML5 semantic elements."""
        tag = element.name.lower()
        
        html5_mapping = {
            'header': ('page_header', 0.9),
            'nav': ('navigation_menu', 0.9),
            'main': ('main_content', 0.95),
            'article': ('article_content', 0.9),
            'section': ('content_section', 0.8),
            'aside': ('sidebar_content', 0.7),
            'footer': ('page_footer', 0.8),
            'h1': ('primary_heading', 1.0),
            'h2': ('secondary_heading', 0.9),
            'h3': ('tertiary_heading', 0.8),
            'figure': ('media_content', 0.7),
            'time': ('date_time', 0.8)
        }
        
        if tag in html5_mapping:
            role, score = html5_mapping[tag]
            return score, role
        
        return 0.0, 'generic_content'
    
    def _analyze_interactive_elements(self, element, text: str) -> Tuple[float, str]:
        """Analyze interactive elements."""
        tag = element.name.lower()
        
        if tag == 'button' or element.get('type') == 'button':
            return 0.9, 'action_button'
        elif tag == 'a' and element.get('href'):
            return 0.8, 'navigation_link'
        elif tag in ['input', 'textarea', 'select']:
            return 0.7, 'form_field'
        elif element.get('onclick') or element.get('data-toggle'):
            return 0.6, 'interactive_element'
        
        return 0.0, 'generic_content'
    
    def extract_interactive_elements(self, element) -> List[Dict[str, str]]:
        """Extract interactive elements from a content block."""
        interactive_elements = []
        
        # Find all interactive elements
        buttons = element.find_all(['button', 'input'], type=['button', 'submit'])
        links = element.find_all('a', href=True)
        forms = element.find_all('form')
        
        for btn in buttons:
            interactive_elements.append({
                'type': 'button',
                'text': btn.get_text(strip=True),
                'action': btn.get('onclick', ''),
                'form_action': btn.get('formaction', '')
            })
        
        for link in links:
            interactive_elements.append({
                'type': 'link',
                'text': link.get_text(strip=True),
                'href': link.get('href', ''),
                'target': link.get('target', '')
            })
        
        for form in forms:
            interactive_elements.append({
                'type': 'form',
                'action': form.get('action', ''),
                'method': form.get('method', 'GET'),
                'fields': str(len(form.find_all(['input', 'textarea', 'select'])))
            })
        
        return interactive_elements
    
    def classify_content_type(self, content_blocks: List[Any]) -> List[str]:
        """Enhanced content type classification."""
        classifications = []
        
        # Aggregate all roles and content
        all_roles = [block.semantic_role for block in content_blocks if hasattr(block, 'semantic_role')]
        all_text = ' '.join([block.text_content for block in content_blocks if hasattr(block, 'text_content')])
        
        role_counts = Counter(all_roles)
        
        # Enhanced classification logic
        if role_counts.get('project_card', 0) > 0 or 'مشروع' in all_text or 'مشاريع' in all_text:
            classifications.append('project_listing')
        
        if role_counts.get('job_listing', 0) > 0:
            classifications.append('job_board')
        
        if role_counts.get('user_profile', 0) > 0:
            classifications.append('social_platform')
        
        if role_counts.get('price_display', 0) > 2:
            classifications.append('e_commerce')
        
        if role_counts.get('article_content', 0) > 0:
            classifications.append('content_site')
        
        if role_counts.get('form_field', 0) > 3:
            classifications.append('form_heavy')
        
        return classifications if classifications else ['general_website']

class EnhancedIntelligentScraper:
    """Enhanced scraper with improved content understanding."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.analyzer = AIContentAnalyzer()
        self.enhanced_analyzer = EnhancedSemanticAnalyzer()
        self.visited_urls = set()
    
    def scrape_and_analyze(self, url: str, options: Dict[str, Any] = None) -> PageAnalysis:
        """Enhanced scraping with improved content analysis."""
        start_time = time.time()
        
        if options is None:
            options = {}
        
        logger.info(f"Starting enhanced analysis of: {url}")
        
        # Fetch page content
        soup = self._get_page_content(url)
        if not soup:
            return PageAnalysis(url=url, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Extract basic page information
        analysis = PageAnalysis(
            url=url,
            title=self._extract_title(soup),
            meta_description=self._extract_meta_description(soup),
            language=self._detect_language(soup),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Extract and analyze content blocks with enhanced understanding
        content_blocks = self._extract_meaningful_content_blocks(soup)
        for i, block in enumerate(content_blocks):
            self._perform_enhanced_analysis(block, soup, i, len(content_blocks), options)
        
        analysis.main_content_blocks = content_blocks
        
        # Extract additional elements if requested
        if options.get("extract_links", False):
            analysis.links = self._extract_links(soup, url)
        
        if options.get("extract_images", False):
            analysis.images = self._extract_images(soup, url)
        
        # Perform enhanced page-level analysis
        self._perform_enhanced_page_analysis(analysis, options)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        analysis.processing_time_ms = int(processing_time)
        
        logger.info(f"Enhanced analysis completed in {processing_time:.2f}ms")
        return analysis
    
    def _extract_meaningful_content_blocks(self, soup: BeautifulSoup) -> List[ContentBlock]:
        """Extract meaningful content blocks with enhanced detection."""
        content_blocks = []
        block_counter = 0
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Prioritize meaningful elements
        priority_selectors = [
            # Cards and listings
            '[class*="card"]', '[class*="item"]', '[class*="post"]', '[class*="listing"]',
            '[class*="project"]', '[class*="job"]', '[class*="product"]',
            # Interactive elements
            'button', '[role="button"]', 'a[href]', 'form',
            # Content sections
            'article', 'section', 'main', 'aside',
            # Headers and important text
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            # Tables and lists
            'table', 'ul', 'ol',
            # Divs with meaningful classes
            'div[class*="content"]', 'div[class*="wrapper"]', 'div[class*="container"]'
        ]
        
        processed_elements = set()
        
        # Extract elements by priority
        for selector in priority_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    if id(element) in processed_elements:
                        continue
                    
                    text_content = element.get_text(separator=" ", strip=True)
                    
                    # Skip empty or very short content
                    if len(text_content) < 15:
                        continue
                    
                    # Skip if this is a child of an already processed element
                    if self._is_child_of_processed(element, processed_elements):
                        continue
                    
                    processed_elements.add(id(element))
                    block_counter += 1
                    
                    # Create enhanced content context
                    context = self._create_enhanced_context(element, soup)
                    
                    # Enhanced semantic role detection
                    semantic_role, confidence = self.enhanced_analyzer.analyze_semantic_role(
                        text_content, element, context
                    )
                    
                    # Extract interactive elements
                    interactive_elements = self.enhanced_analyzer.extract_interactive_elements(element)
                    
                    content_block = ContentBlock(
                        block_id=f"block_{block_counter}",
                        semantic_role=semantic_role,
                        text_content=text_content,
                        html_snippet=self._get_clean_html_snippet(element),
                        xpath=self._get_xpath(element),
                        attributes=dict(element.attrs) if element.attrs else {},
                        interactive_elements=interactive_elements
                    )
                    
                    # Enhanced metadata
                    content_block.metadata = {
                        "semantic_confidence": confidence,
                        "element_tag": element.name,
                        "element_classes": ' '.join(element.get('class', [])),
                        "element_id": element.get('id', ''),
                        "interactive_count": len(interactive_elements),
                        "context_score": context.interactive_score
                    }
                    
                    content_blocks.append(content_block)
                    
            except Exception as e:
                logger.warning(f"Error processing selector {selector}: {e}")
        
        return content_blocks
    
    def _is_child_of_processed(self, element, processed_elements: set) -> bool:
        """Check if element is a child of already processed elements."""
        for parent in element.parents:
            if id(parent) in processed_elements:
                return True
        return False
    
    def _create_enhanced_context(self, element, soup: BeautifulSoup) -> ContentContext:
        """Create enhanced context with interactive scoring."""
        # Calculate position in page
        all_elements = soup.find_all()
        try:
            element_index = all_elements.index(element)
            position_in_page = element_index / len(all_elements)
        except ValueError:
            position_in_page = 0.5
        
        # Calculate interactive score
        interactive_score = 0.0
        if element.find(['a', 'button', 'input', 'form']):
            interactive_score += 0.5
        if element.get('onclick') or element.get('data-toggle'):
            interactive_score += 0.3
        if 'clickable' in ' '.join(element.get('class', [])).lower():
            interactive_score += 0.2
        
        # Visual prominence based on tag and classes
        visual_prominence = 0.0
        if element.name in ['h1', 'h2', 'h3']:
            visual_prominence = 0.9
        elif element.name in ['h4', 'h5', 'h6']:
            visual_prominence = 0.7
        elif 'highlight' in ' '.join(element.get('class', [])).lower():
            visual_prominence = 0.8
        
        return ContentContext(
            position_in_page=position_in_page,
            visual_prominence=visual_prominence,
            interactive_score=interactive_score,
            text_density=len(element.get_text(strip=True)) / max(1, len(str(element))),
            link_density=len(element.find_all('a')) / max(1, len(element.get_text().split()))
        )
    
    def _perform_enhanced_analysis(self, block: ContentBlock, soup: BeautifulSoup, 
                                 position: int, total_blocks: int, options: Dict[str, Any]):
        """Perform enhanced analysis on content blocks."""
        text = block.text_content
        
        # Enhanced entity extraction
        block.entities = self.analyzer.extract_entities(text)
        
        # Enhanced keyword extraction with context
        block.keywords = self._extract_contextual_keywords(text, block.semantic_role)
        
        # Sentiment analysis
        block.sentiment = self.analyzer.analyze_sentiment(text)
        
        # Enhanced topic extraction
        block.topics = self._extract_contextual_topics(text, block.semantic_role)
        
        # Generate summary if requested
        if options.get("summarize_content", False) and len(text) > 200:
            block.summary = self.analyzer.generate_summary(text, 2)
        
        # Enhanced importance scoring
        block.importance_score = self._calculate_enhanced_importance_score(block, position, total_blocks)
    
    def _extract_contextual_keywords(self, text: str, semantic_role: str) -> List[str]:
        """Extract keywords with context awareness."""
        base_keywords = self.analyzer.extract_keywords(text, 15)
        
        # Filter keywords based on semantic role
        if semantic_role == 'project_card':
            # Prioritize project-related keywords
            project_keywords = [kw for kw in base_keywords if any(term in kw.lower() 
                               for term in ['develop', 'design', 'build', 'create', 'app', 'website', 'system'])]
            return project_keywords[:10] if project_keywords else base_keywords[:10]
        
        elif semantic_role == 'price_display':
            # Prioritize price-related keywords
            price_keywords = [kw for kw in base_keywords if any(term in kw.lower() 
                             for term in ['price', 'cost', 'budget', 'payment', 'fee'])]
            return price_keywords[:5] if price_keywords else base_keywords[:5]
        
        return base_keywords[:10]
    
    def _extract_contextual_topics(self, text: str, semantic_role: str) -> List[str]:
        """Extract topics with context awareness."""
        base_topics = self.analyzer.extract_topics(text, 5)
        
        # Enhance topics based on semantic role
        if semantic_role == 'project_card':
            # Look for technology and skill-related topics
            tech_terms = re.findall(r'\b(?:python|javascript|react|node|php|java|android|ios|web|mobile|api|database)\b', 
                                  text.lower())
            if tech_terms:
                base_topics.extend(list(set(tech_terms))[:3])
        
        return list(set(base_topics))[:5]
    
    def _calculate_enhanced_importance_score(self, block: ContentBlock, position: int, total_blocks: int) -> float:
        """Calculate enhanced importance score."""
        score = 0.0
        
        # Enhanced role-based scoring
        role_scores = {
            'project_card': 1.0,
            'job_listing': 0.95,
            'action_button': 0.9,
            'price_display': 0.85,
            'user_profile': 0.8,
            'content_card': 0.75,
            'primary_heading': 0.9,
            'secondary_heading': 0.7,
            'navigation_item': 0.4,
            'form_field': 0.6,
            'date_time': 0.3
        }
        
        score += role_scores.get(block.semantic_role, 0.4)
        
        # Interactive elements boost
        if block.interactive_elements:
            score += min(0.3, len(block.interactive_elements) * 0.1)
        
        # Context-based adjustments
        if hasattr(block, 'metadata'):
            interactive_score = block.metadata.get('context_score', 0)
            score += interactive_score * 0.2
        
        # Position-based scoring (top elements are often more important)
        position_factor = 1.0 - (position / max(1, total_blocks - 1)) * 0.2
        score *= position_factor
        
        # Content quality factors
        text_length = len(block.text_content)
        if 30 <= text_length <= 300:
            score += 0.15
        elif 300 < text_length <= 1000:
            score += 0.1
        
        return min(1.0, score)
    
    def _perform_enhanced_page_analysis(self, analysis: PageAnalysis, options: Dict[str, Any]):
        """Perform enhanced page-level analysis."""
        content_blocks = analysis.main_content_blocks
        
        # Enhanced sentiment analysis
        all_text = ' '.join([block.text_content for block in content_blocks])
        analysis.overall_sentiment = self.analyzer.analyze_sentiment(all_text)
        
        # Enhanced topic extraction
        analysis.page_topics = self._extract_page_topics(content_blocks)
        
        # Enhanced content classification
        analysis.content_categories = self.enhanced_analyzer.classify_content_type(content_blocks)
        
        # Enhanced metadata
        analysis.metadata = {
            "total_interactive_elements": sum(len(block.interactive_elements) for block in content_blocks),
            "semantic_diversity": len(set(block.semantic_role for block in content_blocks)),
            "avg_importance_score": sum(block.importance_score for block in content_blocks) / len(content_blocks) if content_blocks else 0,
            "content_quality_indicators": self._calculate_content_quality_indicators(content_blocks)
        }
        
        # Sort by importance
        content_blocks.sort(key=lambda x: x.importance_score, reverse=True)
    
    def _extract_page_topics(self, content_blocks: List[ContentBlock]) -> List[str]:
        """Extract page-level topics with enhanced logic."""
        all_topics = []
        
        # Collect topics from high-importance blocks
        for block in content_blocks:
            if block.importance_score > 0.6:
                all_topics.extend(block.topics)
        
        # Count and prioritize topics
        topic_counts = Counter(all_topics)
        return [topic for topic, count in topic_counts.most_common(8)]
    
    def _calculate_content_quality_indicators(self, content_blocks: List[ContentBlock]) -> Dict[str, Any]:
        """Calculate content quality indicators."""
        if not content_blocks:
            return {}
        
        return {
            "has_interactive_elements": any(block.interactive_elements for block in content_blocks),
            "content_depth": sum(len(block.text_content) for block in content_blocks),
            "semantic_richness": len(set(block.semantic_role for block in content_blocks)),
            "entity_richness": sum(len(block.entities) for block in content_blocks),
            "structured_content": sum(1 for block in content_blocks if block.semantic_role != 'generic_content')
        }
    
    # Utility methods from base class
    def _get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse page content."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return None
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        return None
    
    def _detect_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Detect page language."""
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            return html_tag['lang']
        
        meta_lang = soup.find('meta', attrs={'http-equiv': 'content-language'})
        if meta_lang and meta_lang.get('content'):
            return meta_lang['content']
        
        return 'en'
    
    def _get_clean_html_snippet(self, element) -> str:
        """Get a clean HTML snippet for the element."""
        html_str = str(element)
        
        if len(html_str) > 500:
            html_str = html_str[:500] + "..."
        
        html_str = re.sub(r'\s+', ' ', html_str)
        return html_str
    
    def _get_xpath(self, element) -> str:
        """Generate XPath for an element."""
        components = []
        child = element if element.name else element.parent
        for parent in child.parents:
            siblings = parent.find_all(child.name, recursive=False)
            if len(siblings) == 1:
                components.append(child.name)
            else:
                try:
                    index = siblings.index(child) + 1
                    components.append(f"{child.name}[{index}]")
                except ValueError:
                    components.append(child.name)
            child = parent
        components.reverse()
        return '/' + '/'.join(components)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            absolute_url = urljoin(base_url, href)
            
            links.append({
                'url': absolute_url,
                'text': text,
                'title': link.get('title', ''),
                'rel': ' '.join(link.get('rel', []))
            })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all images from the page."""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(base_url, src)
            
            images.append({
                'url': absolute_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            })
        
        return images
    
    def save_enhanced_analysis(self, analysis: PageAnalysis, filepath: str):
        """Save enhanced analysis to JSON file."""
        try:
            analysis_dict = self._serialize_analysis(analysis)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(analysis_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Enhanced analysis saved to {filepath}")
            
            # Also save a summary file
            summary_file = filepath.replace(".json", "_summary.json")
            summary = self._create_analysis_summary(analysis)
            
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis summary saved to {summary_file}")
        
        except Exception as e:
            logger.error(f"Error saving enhanced analysis to {filepath}: {e}")
    
    def _serialize_analysis(self, analysis: PageAnalysis) -> Dict[str, Any]:
        """Convert analysis to JSON-serializable dictionary."""
        result = {
            "url": analysis.url,
            "title": analysis.title,
            "meta_description": analysis.meta_description,
            "language": analysis.language,
            "timestamp": analysis.timestamp,
            "processing_time_ms": analysis.processing_time_ms,
            "metadata": analysis.metadata,
            "overall_sentiment": analysis.overall_sentiment,
            "page_topics": analysis.page_topics,
            "content_categories": analysis.content_categories,
            "content_blocks": [],
            "links": analysis.links,
            "images": analysis.images
        }
        
        # Serialize content blocks
        for block in analysis.main_content_blocks:
            block_dict = {
                "block_id": block.block_id,
                "semantic_role": block.semantic_role,
                "text_content": block.text_content,
                "importance_score": block.importance_score,
                "xpath": block.xpath,
                "attributes": block.attributes,
                "keywords": block.keywords,
                "topics": block.topics,
                "sentiment": block.sentiment,
                "summary": block.summary,
                "interactive_elements": block.interactive_elements,
                "metadata": block.metadata,
                "entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "start_char": entity.start_char,
                        "end_char": entity.end_char,
                        "confidence": entity.confidence
                    }
                    for entity in block.entities
                ]
            }
            
            if block.html_snippet and len(block.html_snippet) < 1000:
                block_dict["html_snippet"] = block.html_snippet
            
            result["content_blocks"].append(block_dict)
        
        return result
    
    def _create_analysis_summary(self, analysis: PageAnalysis) -> Dict[str, Any]:
        """Create a concise summary of the analysis."""
        return {
            "url": analysis.url,
            "title": analysis.title,
            "main_topics": analysis.page_topics[:5],
            "content_categories": analysis.content_categories,
            "overall_sentiment": analysis.overall_sentiment.get("overall", "neutral"),
            "total_content_blocks": len(analysis.main_content_blocks),
            "interactive_elements_count": analysis.metadata.get("total_interactive_elements", 0),
            "semantic_diversity": analysis.metadata.get("semantic_diversity", 0),
            "content_quality": analysis.metadata.get("content_quality_indicators", {}),
            "top_content_blocks": [
                {
                    "role": block.semantic_role,
                    "text_preview": block.text_content[:100] + ("..." if len(block.text_content) > 100 else ""),
                    "importance_score": block.importance_score,
                    "interactive_elements": len(block.interactive_elements)
                }
                for block in sorted(analysis.main_content_blocks, key=lambda b: b.importance_score, reverse=True)[:5]
            ],
            "processing_time_ms": analysis.processing_time_ms,
            "timestamp": analysis.timestamp
        }

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Intelligent Web Scraper with Dynamic Content Understanding"
    )
    parser.add_argument("url", type=str, help="URL to scrape and analyze")
    parser.add_argument("-o", "--output", type=str, default="enhanced_analysis.json",
                        help="Output JSON file (default: enhanced_analysis.json)")
    parser.add_argument("--extract-links", action="store_true",
                        help="Extract all links")
    parser.add_argument("--extract-images", action="store_true",
                        help="Extract all images")
    parser.add_argument("--summarize", action="store_true",
                        help="Generate content summaries")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    scraper = EnhancedIntelligentScraper()
    options = {
        "extract_links": args.extract_links,
        "extract_images": args.extract_images,
        "summarize_content": args.summarize
    }

    try:
        analysis = scraper.scrape_and_analyze(args.url, options)
        scraper.save_enhanced_analysis(analysis, args.output)

        print(f"Enhanced analysis complete! Results saved to: {args.output}")
        print("\n=== ENHANCED ANALYSIS SUMMARY ===")
        print(f"Title: {analysis.title}")
        print(f"Content Categories: {', '.join(analysis.content_categories)}")
        print(f"Main Topics: {', '.join(analysis.page_topics[:5])}")
        print(f"Overall Sentiment: {analysis.overall_sentiment.get('overall', 'neutral')}")
        print(f"Total Content Blocks: {len(analysis.main_content_blocks)}")
        print(f"Interactive Elements: {analysis.metadata.get('total_interactive_elements', 0)}")
        print(f"Semantic Diversity: {analysis.metadata.get('semantic_diversity', 0)}")
        print(f"Processing Time: {analysis.processing_time_ms}ms")
        
        print("\n=== TOP CONTENT BLOCKS ===")
        for i, block in enumerate(sorted(analysis.main_content_blocks, key=lambda b: b.importance_score, reverse=True)[:3]):
            print(f"{i+1}. {block.semantic_role} (score: {block.importance_score:.2f})")
            print(f"   Text: {block.text_content[:80]}...")
            if block.interactive_elements:
                print(f"   Interactive: {len(block.interactive_elements)} elements")
            if block.entities:
                print(f"   Entities: {', '.join([f'{e.text}({e.label})' for e in block.entities[:2]])}")

    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

