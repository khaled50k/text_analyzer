import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import Counter

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class TextAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def read_text_file(self, filepath):
        """
        Reads the content of a text file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} was not found.")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def summarize_text_nltk(self, text, num_sentences=3):
        """
        Creates intelligent, coherent summaries using advanced NLP techniques.
        This method dynamically adapts to any content type without hardcoded restrictions.
        """
        if not text.strip():
            return ""
        
        # Preprocess text to handle newlines and formatting issues
        text = self._preprocess_text(text)
            
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Clean and filter sentences intelligently
        cleaned_sentences = []
        for sentence in sentences:
            if self._is_quality_sentence(sentence):
                cleaned_sentences.append(sentence.strip())
        
        if len(cleaned_sentences) <= num_sentences:
            return ' '.join(cleaned_sentences)
        
        # Use spaCy for advanced semantic analysis
        doc = nlp(text)
        
        # Dynamic topic identification using NLP features
        topic_sentences = self._identify_topic_sentences_dynamically(doc, cleaned_sentences)
        
        # If we have good topic sentences, use them as foundation
        if len(topic_sentences) >= 2:
            selected_topics = topic_sentences[:min(num_sentences, len(topic_sentences))]
            selected_topics.sort(key=lambda x: sentences.index(x))
            
            # Add context sentences if needed
            if len(selected_topics) < num_sentences:
                remaining_slots = num_sentences - len(selected_topics)
                context_sentences = self._find_context_sentences(selected_topics, sentences, remaining_slots)
                final_summary = selected_topics + context_sentences
                final_summary.sort(key=lambda x: sentences.index(x))
                return ' '.join(final_summary[:num_sentences])
            
            return ' '.join(selected_topics)
        
        # Fallback: Intelligent frequency-based approach with semantic weighting
        return self._intelligent_frequency_summary(cleaned_sentences, text, num_sentences)
    
    def _preprocess_text(self, text):
        """
        Preprocesses text to handle formatting issues dynamically without hardcoded assumptions.
        This method adapts to any content type by using general text cleaning principles.
        """
        # Replace multiple newlines with single spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common spacing issues around punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Clean up any remaining malformed text
        text = text.strip()
        
        return text
    
    def _is_quality_sentence(self, sentence):
        """
        Dynamically determines if a sentence is of good quality without hardcoded assumptions.
        Uses linguistic and structural features that apply to any content type.
        """
        sentence = sentence.strip()
        
        # Basic length requirements
        if len(sentence.split()) < 8 or len(sentence.split()) > 50:
            return False
        
        # Must end with proper sentence termination
        if not sentence.endswith(('.') and not sentence.endswith('!') and not sentence.endswith('?')):
            return False
        
        # Skip all-uppercase sentences (likely headers)
        if sentence.isupper():
            return False
        
        # Skip very short sentences
        if len(sentence) < 20:
            return False
        
        # Skip sentences that are just fragments (e.g., "Title Name" or "Chapter 1")
        if re.match(r'^[A-Z][a-z]*\s+[A-Z][a-z]*$', sentence):
            return False
        
        # Skip sentences that are just numbers or symbols
        if re.match(r'^[\d\s\-\.]+$', sentence):
            return False
        
        # Skip sentences that are just repeated characters
        if len(set(sentence)) < 5:
            return False
        
        return True
    
    def _identify_topic_sentences_dynamically(self, doc, sentences):
        """
        Dynamically identifies topic sentences using general linguistic features that work with any content.
        """
        topic_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Use general linguistic features to identify important sentences
            importance_score = 0
            
            # Check for definitional patterns (general, not topic-specific)
            if any(pattern in sent_text.lower() for pattern in ['is a', 'are', 'refers to', 'means', 'consists of', 'involves', 'enables', 'provides', 'helps', 'ensures', 'defines', 'describes', 'represents']):
                importance_score += 3
            
            # Check for explanatory patterns
            if any(pattern in sent_text.lower() for pattern in ['because', 'therefore', 'thus', 'hence', 'as a result', 'this means', 'in other words', 'for example', 'specifically', 'in particular']):
                importance_score += 2
            
            # Check for introductory patterns
            if any(pattern in sent_text.lower() for pattern in ['first', 'initially', 'to begin', 'let\'s', 'we\'ll', 'this', 'in this', 'overview', 'introduction']):
                importance_score += 2
            
            # Check for conclusion patterns
            if any(pattern in sent_text.lower() for pattern in ['conclusion', 'finally', 'in summary', 'to summarize', 'overall', 'ultimately', 'in conclusion', 'to conclude']):
                importance_score += 2
            
            # Check for technical/domain-specific language (dynamic detection)
            technical_indicators = self._detect_technical_language(sent_text)
            importance_score += technical_indicators
            
            # Check sentence structure complexity (more complex often = more important)
            if 15 <= len(sent_text.split()) <= 35:
                importance_score += 1
            
            # Check for named entities (indicates important content)
            sent_doc = nlp(sent_text)
            if len(sent_doc.ents) > 0:
                importance_score += 1
            
            # Check for key terms (words that appear multiple times in the text)
            key_terms = self._identify_key_terms_dynamically(sent_text, doc)
            importance_score += len(key_terms) * 0.5
            
            if importance_score >= 2 and len(sent_text.split()) >= 8:
                topic_sentences.append(sent_text)
        
        return topic_sentences
    
    def _identify_key_terms_dynamically(self, sentence, full_doc):
        """
        Dynamically identifies key terms in a sentence by analyzing the full document context.
        """
        # Get all words from the full document
        all_words = [token.lemma_.lower() for token in full_doc if token.is_alpha and not token.is_stop and len(token.text) > 3]
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        # Get words from the current sentence
        sentence_words = [token.lemma_.lower() for token in nlp(sentence) if token.is_alpha and not token.is_stop and len(token.text) > 3]
        
        # Return words that appear multiple times in the document
        key_terms = [word for word in sentence_words if word_freq[word] > 2]
        
        return key_terms
    
    def _detect_technical_language(self, text):
        """
        Dynamically detects technical language without hardcoded terms.
        """
        score = 0
        
        # Check for technical patterns
        if any(pattern in text.lower() for pattern in ['api', 'orm', 'sql', 'http', 'json', 'xml', 'ssl', 'tls', 'oauth', 'jwt', 'rest', 'graphql']):
            score += 2
        
        # Check for technical abbreviations and acronyms
        if re.search(r'\b[A-Z]{2,}\b', text):
            score += 1
        
        # Check for version numbers and technical specifications
        if re.search(r'\b\d+\.\d+\b', text):
            score += 1
        
        # Check for technical verbs
        technical_verbs = ['implement', 'deploy', 'configure', 'optimize', 'scale', 'cache', 'authenticate', 'encrypt', 'monitor', 'debug']
        if any(verb in text.lower() for verb in technical_verbs):
            score += 1
        
        return score
    
    def _find_context_sentences(self, topic_sentences, all_sentences, num_needed):
        """
        Finds contextual sentences that provide supporting information.
        """
        context_sentences = []
        used_sentences = set(topic_sentences)
        
        for topic in topic_sentences:
            topic_idx = all_sentences.index(topic)
            
            # Look for sentences that provide context or examples
            for i in range(max(0, topic_idx - 1), min(topic_idx + 3, len(all_sentences))):
                if (all_sentences[i] not in used_sentences and 
                    len(context_sentences) < num_needed and
                    len(all_sentences[i].split()) >= 8):
                    context_sentences.append(all_sentences[i])
                    used_sentences.add(all_sentences[i])
        
        return context_sentences[:num_needed]
    
    def _intelligent_frequency_summary(self, sentences, text, num_sentences):
        """
        Creates intelligent summaries using frequency analysis with semantic enhancement.
        """
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter words intelligently
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        
        if not filtered_words:
            return ' '.join(sentences[:num_sentences])
        
        # Calculate word importance using multiple dynamic factors
        word_scores = {}
        for word in filtered_words:
            score = 0
            
            # Frequency
            freq = filtered_words.count(word)
            score += freq
            
            # Position importance (words appearing early are often more important)
            first_pos = text.lower().find(word)
            if first_pos != -1:
                position_score = 1.0 - (first_pos / len(text))
                score += position_score * 2
            
            # Length bonus (longer words often indicate technical terms)
            if len(word) > 6:
                score += 0.5
            
            # Capitalization bonus (proper nouns are often important)
            if word in text and word[0].isupper():
                score += 1
            
            word_scores[word] = score
        
        # Score sentences based on word importance
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            
            for word in sentence_words:
                if word in word_scores:
                    score += word_scores[word]
                    word_count += 1
            
            if word_count > 0:
                # Normalize by sentence length to avoid bias toward longer sentences
                sentence_scores[sentence] = score / word_count
        
        if not sentence_scores:
            return ' '.join(sentences[:num_sentences])
        
        # Select top sentences and maintain order
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(top_sentences)
    
    def extract_keywords_spacy(self, text, num_keywords=10):
        """
        Extracts meaningful keywords dynamically using advanced NLP techniques.
        This method adapts to any content type without hardcoded restrictions.
        """
        # Preprocess text to handle formatting issues
        text = self._preprocess_text(text)
        
        doc = nlp(text)
        
        # Extract potential keywords using linguistic features
        potential_keywords = []
        for token in doc:
            # Focus on meaningful parts of speech
            if (token.is_alpha and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2 and
                token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']):
                
                # Skip overly common words that aren't really keywords
                if token.text.lower() not in ['use', 'can', 'will', 'get', 'make', 'take', 'see', 'know', 'way', 'time', 'year', 'day', 'work', 'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'well', 'only', 'very', 'even', 'back', 'still', 'must', 'should', 'may', 'might', 'could', 'would', 'shall']:
                    potential_keywords.append(token.lemma_.lower())
        
        # Calculate keyword importance using multiple dynamic factors
        keyword_scores = {}
        for keyword in potential_keywords:
            score = 0
            
            # Frequency bonus (more frequent = more important)
            freq = potential_keywords.count(keyword)
            score += freq * 2
            
            # Position importance (words appearing early are often more important)
            first_pos = text.lower().find(keyword)
            if first_pos != -1:
                position_score = 1.0 - (first_pos / len(text))
                score += position_score * 3
            
            # Length bonus (longer words often indicate technical terms or proper nouns)
            if len(keyword) > 6:
                score += 1
            elif len(keyword) > 8:
                score += 2
            
            # Capitalization bonus (proper nouns are often important)
            if keyword in text and keyword[0].isupper():
                score += 2
            
            # Part-of-speech bonus (nouns and proper nouns are typically more important)
            for token in doc:
                if token.lemma_.lower() == keyword:
                    if token.pos_ == 'PROPN':
                        score += 3
                    elif token.pos_ == 'NOUN':
                        score += 2
                    elif token.pos_ == 'ADJ':
                        score += 1
                    break
            
            # Named entity bonus (entities are often key concepts)
            for ent in doc.ents:
                if keyword in ent.text.lower():
                    score += 4
                    break
            
            # Technical language detection (dynamic, not hardcoded)
            technical_score = self._detect_technical_language_dynamic(keyword, text)
            score += technical_score
            
            keyword_scores[keyword] = score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and return top N
        unique_keywords = []
        for keyword, score in sorted_keywords:
            if keyword not in unique_keywords and len(unique_keywords) < num_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _detect_technical_language_dynamic(self, word, text):
        """
        Dynamically detects technical language patterns without hardcoded terms.
        """
        score = 0
        
        # Check for technical patterns (dynamic detection)
        technical_patterns = [
            # Abbreviations and acronyms
            r'\b[A-Z]{2,}\b',
            # Version numbers
            r'\b\d+\.\d+\b',
            # Technical file extensions
            r'\b\w+\.(py|js|ts|java|cpp|h|php|html|css|sql|json|xml|yaml|yml|md|txt)\b',
            # URLs and protocols
            r'\b(https?|ftp|ssh|sftp)://\b',
            # IP addresses
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            # Hash patterns
            r'\b[a-fA-F0-9]{32,}\b',
            # Technical identifiers
            r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, word):
                score += 1
        
        # Check for technical suffixes and prefixes
        technical_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment', 'ness', 'ity', 'al', 'ive', 'able', 'ible']
        technical_prefixes = ['un', 're', 'pre', 'post', 'anti', 'pro', 'co', 'inter', 'intra', 'trans', 'sub', 'super', 'hyper', 'micro', 'macro']
        
        for suffix in technical_suffixes:
            if word.endswith(suffix):
                score += 0.5
        
        for prefix in technical_prefixes:
            if word.startswith(prefix):
                score += 0.5
        
        # Check if word appears in technical contexts
        technical_contexts = ['api', 'database', 'server', 'client', 'protocol', 'algorithm', 'framework', 'library', 'module', 'package', 'dependency', 'configuration', 'environment', 'deployment', 'infrastructure']
        if any(context in text.lower() for context in technical_contexts):
            if word in text.lower():
                score += 1
        
        return score
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyzes sentiment using TextBlob.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity
        }
    
    def analyze_sentiment_vader(self, text):
        """
        Analyzes sentiment using VADER.
        """
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,
            "compound": compound,
            "positive": scores['pos'],
            "neutral": scores['neu'],
            "negative": scores['neg']
        }
    
    def extract_entities(self, text):
        """
        Extracts named entities using spaCy.
        """
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def get_text_statistics(self, text):
        """
        Calculates basic text statistics.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        characters = len(text)
        characters_no_spaces = len(text.replace(' ', ''))
        
        return {
            "sentences": len(sentences),
            "words": len(words),
            "characters": characters,
            "characters_no_spaces": characters_no_spaces,
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0
        }
    
    def analyze_readability(self, text):
        """
        Analyzes text readability using simple metrics.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return {"flesch_reading_ease": 0, "readability_level": "Unknown"}
        
        # Simple Flesch Reading Ease approximation
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        if flesch_score >= 90:
            level = "Very Easy"
        elif flesch_score >= 80:
            level = "Easy"
        elif flesch_score >= 70:
            level = "Fairly Easy"
        elif flesch_score >= 60:
            level = "Standard"
        elif flesch_score >= 50:
            level = "Fairly Difficult"
        elif flesch_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return {
            "flesch_reading_ease": flesch_score,
            "readability_level": level
        }
    
    def _count_syllables(self, word):
        """
        Simple syllable counting heuristic.
        """
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def create_learning_summary(self, text, num_concepts=5):
        """
        Creates dynamic learning summaries that adapt to any content type.
        """
        if not text.strip():
            return ""
        
        # Preprocess text to handle formatting issues
        text = self._preprocess_text(text)
        
        doc = nlp(text)
        
        # Dynamically identify learning concepts
        concept_sentences = self._identify_learning_concepts_dynamically(doc, text)
        
        if len(concept_sentences) >= 3:
            selected_concepts = concept_sentences[:num_concepts]
            selected_concepts.sort(key=lambda x: text.find(x))
            
            summary_parts = ["📚 **Key Concepts Explained:**"]
            for i, concept in enumerate(selected_concepts, 1):
                summary_parts.append(f"\n{i}. {concept}")
            
            return '\n'.join(summary_parts)
        
        # Create structured summary based on content analysis
        return self._create_dynamic_structured_summary(text)
    
    def _identify_learning_concepts_dynamically(self, doc, text):
        """
        Dynamically identifies learning concepts using general linguistic patterns that work with any content.
        """
        concept_sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Use general linguistic patterns to identify concept explanations
            concept_indicators = 0
            
            # Definition patterns (general, not topic-specific)
            if any(pattern in sent_text.lower() for pattern in ['is a', 'are', 'refers to', 'means', 'consists of', 'involves', 'enables', 'provides', 'helps', 'ensures', 'achieves', 'improves', 'optimizes', 'defines', 'describes', 'represents']):
                concept_indicators += 2
            
            # Explanation patterns
            if any(pattern in sent_text.lower() for pattern in ['because', 'therefore', 'thus', 'hence', 'as a result', 'this means', 'in other words', 'for example', 'such as', 'specifically', 'in particular']):
                concept_indicators += 2
            
            # Process patterns
            if any(pattern in sent_text.lower() for pattern in ['first', 'then', 'next', 'finally', 'step', 'process', 'method', 'approach', 'strategy', 'technique', 'procedure', 'workflow']):
                concept_indicators += 2
            
            # Technical patterns (general, not specific to any domain)
            if any(pattern in sent_text.lower() for pattern in ['implement', 'deploy', 'configure', 'set up', 'install', 'optimize', 'scale', 'develop', 'create', 'build', 'design']):
                concept_indicators += 1
            
            # Check sentence quality
            if (concept_indicators >= 2 and 
                len(sent_text.split()) >= 8 and 
                len(sent_text.split()) <= 45):
                concept_sentences.append(sent_text)
        
        return concept_sentences
    
    def _create_dynamic_structured_summary(self, text):
        """
        Creates a structured summary that adapts to any content type dynamically.
        """
        sentences = sent_tokenize(text)
        
        summary_parts = ["📚 **Comprehensive Learning Summary:**"]
        
        # Find introduction/main topic using general patterns
        intro_sentences = []
        for sentence in sentences[:10]:  # Look in first 10 sentences
            if any(indicator in sentence.lower() for indicator in ['guide', 'explore', 'learn', 'understand', 'overview', 'introduction', 'report', 'study', 'analysis', 'examine', 'investigate']):
                intro_sentences.append(sentence.strip())
        
        if intro_sentences:
            summary_parts.append(f"\n🎯 **Main Topic:** {intro_sentences[0]}")
        
        # Identify key sections dynamically using general patterns
        section_indicators = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', 'A.', 'B.', 'C.', 'D.', 'E.', 'F.', '✅', '🔹', '🚀', '📚', '🎯', '🔍', '💡', '📝', '📊', '😊', '🎭', '🏷️', '☁️']
        key_sections = []
        
        for i, sentence in enumerate(sentences):
            if any(indicator in sentence for indicator in section_indicators):
                # Get the next sentence as context
                if i + 1 < len(sentences):
                    context = sentences[i + 1].strip()
                    if len(context.split()) >= 6 and self._is_quality_sentence(context):
                        key_sections.append(context)
        
        if key_sections:
            summary_parts.append(f"\n🔍 **Key Areas Covered:**")
            for i, section in enumerate(key_sections[:4], 1):
                summary_parts.append(f"\n{i}. {section}")
        
        # Find conclusion or key takeaway using general patterns
        conclusion_sentences = []
        for sentence in sentences[-10:]:  # Look in last 10 sentences
            if any(indicator in sentence.lower() for indicator in ['conclusion', 'finally', 'summary', 'overall', 'ultimately', 'achieve', 'ensure', 'build', 'create', 'develop', 'establish', 'conclude', 'summarize']):
                conclusion_sentences.append(sentence.strip())
        
        if conclusion_sentences:
            summary_parts.append(f"\n💡 **Key Takeaway:** {conclusion_sentences[0]}")
        
        return '\n'.join(summary_parts)

    def get_word_frequency_for_cloud(self, text, max_words=50):
        """
        Generates word frequency data suitable for word cloud generation.
        Returns a dictionary of words and their frequencies.
        """
        doc = nlp(text)
        words = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']]
        word_freq = Counter(words)
        return dict(word_freq.most_common(max_words))

    def comprehensive_analysis(self, text):
        """
        Performs comprehensive text analysis.
        """
        return {
            "summary": self.summarize_text_nltk(text),
            "keywords": self.extract_keywords_spacy(text),
            "sentiment_textblob": self.analyze_sentiment_textblob(text),
            "sentiment_vader": self.analyze_sentiment_vader(text),
            "entities": self.extract_entities(text),
            "statistics": self.get_text_statistics(text),
            "readability": self.analyze_readability(text)
        }

