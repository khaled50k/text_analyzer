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
        Summarizes text using NLTK by extracting the most important sentences.
        Improved version for better coherence and readability.
        """
        if not text.strip():
            return ""
            
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip very short sentences, headers, and bullet points
            if (len(sentence.split()) >= 5 and 
                len(sentence.split()) <= 50 and
                not sentence.startswith(('✅', '🔹', '🚀', '🔹', 'A.', 'B.', 'C.', '1.', '2.', '3.', '4.', '5.', '6.')) and
                not sentence.isupper() and
                sentence.endswith(('.') or sentence.endswith('!') or sentence.endswith('?'))):
                cleaned_sentences.append(sentence)
        
        if len(cleaned_sentences) <= num_sentences:
            return ' '.join(cleaned_sentences)
        
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter out stop words and non-alphabetic tokens
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Calculate word frequency
        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        # Normalize frequencies
        if not word_frequencies:
            return ' '.join(cleaned_sentences[:num_sentences])
            
        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for sentence in cleaned_sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            for word in sentence_words:
                if word in word_frequencies.keys():
                    score += word_frequencies[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count  # Average score
        
        # If no sentences scored, return first few cleaned sentences
        if not sentence_scores:
            return ' '.join(cleaned_sentences[:num_sentences])
        
        # Sort sentences by score and return the top ones
        summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        
        # Sort by original position to maintain logical flow
        summarized_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(summarized_sentences)
    
    def extract_keywords_spacy(self, text, num_keywords=10):
        """
        Extracts keywords from text using spaCy.
        """
        doc = nlp(text)
        keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
        # Simple frequency-based keyword selection
        keyword_freq = Counter(keywords)
        return [word for word, freq in keyword_freq.most_common(num_keywords)]
    
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
        Creates a learning-focused summary that explains key concepts and their relationships.
        This helps readers understand the main ideas rather than just extracting sentences.
        """
        if not text.strip():
            return ""
        
        # Extract key concepts and their explanations
        doc = nlp(text)
        
        # Find sentences that contain definitions, explanations, or key concepts
        concept_sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Look for sentences that explain concepts
            if any(keyword in sent_text.lower() for keyword in ['means', 'refers to', 'is a', 'are', 'consists of', 'involves', 'enables', 'provides', 'helps', 'ensures', 'achieve', 'improve', 'optimize']):
                if len(sent_text.split()) >= 8 and len(sent_text.split()) <= 40:
                    concept_sentences.append(sent_text)
        
        # If we don't have enough concept sentences, fall back to regular summary
        if len(concept_sentences) < 2:
            return self.summarize_text_nltk(text, num_concepts)
        
        # Select the most relevant concept sentences
        selected_concepts = concept_sentences[:num_concepts]
        
        # Create a structured summary
        summary_parts = []
        summary_parts.append("📚 **Key Concepts Explained:**")
        
        for i, concept in enumerate(selected_concepts, 1):
            summary_parts.append(f"\n{i}. {concept}")
        
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
            # "entities": self.extract_entities(text),
            "statistics": self.get_text_statistics(text),
            "readability": self.analyze_readability(text)
        }

