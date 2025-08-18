import unittest
import os
import sys
import tempfile

# Add parent directory to path to import text_analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_analyzer import TextAnalyzer

class TestTextAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = TextAnalyzer()
        self.sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        unlike the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of 'intelligent agents': 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals. Colloquially, the term 
        'artificial intelligence' is often used to describe machines (or computers) 
        that mimic 'cognitive' functions that humans associate with the human mind, 
        such as 'learning' and 'problem-solving'.
        """
        
    def test_read_text_file(self):
        """Test reading text from a file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(self.sample_text)
            temp_file_path = f.name
        
        try:
            # Test reading the file
            content = self.analyzer.read_text_file(temp_file_path)
            self.assertEqual(content, self.sample_text)
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_read_nonexistent_file(self):
        """Test reading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.read_text_file('/nonexistent/file.txt')
    
    def test_summarize_text_nltk(self):
        """Test text summarization functionality."""
        summary = self.analyzer.summarize_text_nltk(self.sample_text, num_sentences=2)
        
        # Check that summary is not empty
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary.strip()), 0)
        
        # Check that summary is shorter than original text
        self.assertLess(len(summary), len(self.sample_text))
    
    def test_extract_keywords_spacy(self):
        """Test keyword extraction functionality."""
        keywords = self.analyzer.extract_keywords_spacy(self.sample_text, num_keywords=5)
        
        # Check that keywords is a list
        self.assertIsInstance(keywords, list)
        
        # Check that we get some keywords
        self.assertGreater(len(keywords), 0)
        
        # Check that all keywords are strings
        for keyword in keywords:
            self.assertIsInstance(keyword, str)
    
    def test_analyze_sentiment_textblob(self):
        """Test sentiment analysis with TextBlob."""
        sentiment = self.analyzer.analyze_sentiment_textblob(self.sample_text)
        
        # Check that sentiment analysis returns expected structure
        self.assertIsInstance(sentiment, dict)
        self.assertIn('sentiment', sentiment)
        self.assertIn('polarity', sentiment)
        self.assertIn('subjectivity', sentiment)
        
        # Check that sentiment is one of expected values
        self.assertIn(sentiment['sentiment'], ['Positive', 'Negative', 'Neutral'])
        
        # Check that polarity and subjectivity are floats
        self.assertIsInstance(sentiment['polarity'], float)
        self.assertIsInstance(sentiment['subjectivity'], float)
    
    def test_analyze_sentiment_vader(self):
        """Test sentiment analysis with VADER."""
        sentiment = self.analyzer.analyze_sentiment_vader(self.sample_text)
        
        # Check that sentiment analysis returns expected structure
        self.assertIsInstance(sentiment, dict)
        self.assertIn('sentiment', sentiment)
        self.assertIn('compound', sentiment)
        self.assertIn('positive', sentiment)
        self.assertIn('neutral', sentiment)
        self.assertIn('negative', sentiment)
        
        # Check that sentiment is one of expected values
        self.assertIn(sentiment['sentiment'], ['Positive', 'Negative', 'Neutral'])
        
        # Check that all scores are floats
        for key in ['compound', 'positive', 'neutral', 'negative']:
            self.assertIsInstance(sentiment[key], float)
    
    def test_extract_entities(self):
        """Test named entity extraction."""
        entities = self.analyzer.extract_entities(self.sample_text)
        
        # Check that entities is a list
        self.assertIsInstance(entities, list)
        
        # Check that each entity is a tuple with text and label
        for entity in entities:
            self.assertIsInstance(entity, tuple)
            self.assertEqual(len(entity), 2)
            self.assertIsInstance(entity[0], str)  # entity text
            self.assertIsInstance(entity[1], str)  # entity label
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        stats = self.analyzer.get_text_statistics(self.sample_text)
        
        # Check that stats is a dictionary with expected keys
        self.assertIsInstance(stats, dict)
        expected_keys = ['sentences', 'words', 'characters', 'characters_no_spaces', 'avg_words_per_sentence']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that all values are numbers
        for key in expected_keys:
            self.assertIsInstance(stats[key], (int, float))
        
        # Check that values make sense
        self.assertGreater(stats['sentences'], 0)
        self.assertGreater(stats['words'], 0)
        self.assertGreater(stats['characters'], 0)
        self.assertGreater(stats['characters_no_spaces'], 0)
        self.assertGreater(stats['avg_words_per_sentence'], 0)
    
    def test_analyze_readability(self):
        """Test readability analysis."""
        readability = self.analyzer.analyze_readability(self.sample_text)
        
        # Check that readability is a dictionary with expected keys
        self.assertIsInstance(readability, dict)
        self.assertIn('flesch_reading_ease', readability)
        self.assertIn('readability_level', readability)
        
        # Check that flesch_reading_ease is a float
        self.assertIsInstance(readability['flesch_reading_ease'], float)
        
        # Check that readability_level is a string
        self.assertIsInstance(readability['readability_level'], str)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis functionality."""
        analysis = self.analyzer.comprehensive_analysis(self.sample_text)
        
        # Check that analysis is a dictionary with expected keys
        self.assertIsInstance(analysis, dict)
        expected_keys = ['summary', 'keywords', 'sentiment_textblob', 'sentiment_vader', 
                        'entities', 'statistics', 'readability']
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check that each component returns expected types
        self.assertIsInstance(analysis['summary'], str)
        self.assertIsInstance(analysis['keywords'], list)
        self.assertIsInstance(analysis['sentiment_textblob'], dict)
        self.assertIsInstance(analysis['sentiment_vader'], dict)
        self.assertIsInstance(analysis['entities'], list)
        self.assertIsInstance(analysis['statistics'], dict)
        self.assertIsInstance(analysis['readability'], dict)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        empty_text = ""
        
        # Test that empty text doesn't crash the analyzer
        summary = self.analyzer.summarize_text_nltk(empty_text)
        self.assertEqual(summary, "")
        
        keywords = self.analyzer.extract_keywords_spacy(empty_text)
        self.assertIsInstance(keywords, list)
        
        sentiment_tb = self.analyzer.analyze_sentiment_textblob(empty_text)
        self.assertIsInstance(sentiment_tb, dict)
        
        sentiment_vader = self.analyzer.analyze_sentiment_vader(empty_text)
        self.assertIsInstance(sentiment_vader, dict)

if __name__ == '__main__':
    unittest.main()

