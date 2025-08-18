# 🤖 AI Text Analyzer

A powerful and unique Python project that reads text files and provides AI-powered text summarization and analysis using NLTK and spaCy.

## ✨ Features

- **📝 Smart Summarization**: Extract key sentences using advanced NLP algorithms
- **🎯 Keyword Extraction**: Identify the most important terms and concepts
- **😊 Sentiment Analysis**: Analyze emotional tone using multiple AI models (TextBlob & VADER)
- **🏷️ Entity Recognition**: Detect people, places, organizations, and more
- **📊 Text Statistics**: Comprehensive metrics about your text
- **📖 Readability Analysis**: Assess how easy your text is to read
- **🌐 Web Interface**: Beautiful Flask-based web application
- **💻 Command Line Interface**: Powerful CLI for batch processing
- **🔌 API Endpoint**: RESTful API for integration with other applications

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd text_analyzer
   ```

2. **Install required packages**
   ```bash
   pip install nltk spacy textblob vaderSentiment flask
   ```

3. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

## 📖 Usage

### Web Interface

1. **Start the web application**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```

3. **Upload a text file or paste text directly** for analysis

### Command Line Interface

#### Basic Usage

```bash
# Analyze a text file
python cli.py sample.txt -f

# Analyze text directly
python cli.py "Your text here"
```

#### Advanced Options

```bash
# Customize number of summary sentences and keywords
python cli.py sample.txt -f -s 5 -k 15

# Output results in JSON format
python cli.py sample.txt -f --json

# Save results to a file
python cli.py sample.txt -f -o results.json

# Skip specific analysis components
python cli.py sample.txt -f --no-sentiment --no-entities
```

#### CLI Options

- `-f, --file`: Treat input as file path (default: treat as text)
- `-s, --summary`: Number of sentences in summary (default: 3)
- `-k, --keywords`: Number of keywords to extract (default: 10)
- `-o, --output`: Output file path (JSON format)
- `--json`: Output results in JSON format
- `--no-summary`: Skip text summarization
- `--no-sentiment`: Skip sentiment analysis
- `--no-entities`: Skip entity extraction
- `--no-stats`: Skip text statistics
- `--no-readability`: Skip readability analysis

### API Usage

#### Start the Flask application
```bash
python app.py
```

#### Make API requests
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze here"}'
```

### Python Module Usage

```python
from text_analyzer import TextAnalyzer

# Initialize the analyzer
analyzer = TextAnalyzer()

# Read text from file
text = analyzer.read_text_file('sample.txt')

# Perform comprehensive analysis
results = analyzer.comprehensive_analysis(text)

# Access individual components
summary = analyzer.summarize_text_nltk(text, num_sentences=3)
keywords = analyzer.extract_keywords_spacy(text, num_keywords=10)
sentiment = analyzer.analyze_sentiment_textblob(text)
entities = analyzer.extract_entities(text)
stats = analyzer.get_text_statistics(text)
readability = analyzer.analyze_readability(text)
```

## 🏗️ Project Structure

```
text_analyzer/
├── app.py                 # Flask web application
├── cli.py                 # Command-line interface
├── text_analyzer.py       # Core analysis module
├── main.py               # Simple usage example
├── templates/            # HTML templates for web interface
│   ├── base.html
│   ├── index.html
│   └── results.html
├── tests/                # Unit tests
│   └── test_text_analyzer.py
├── sample.txt            # Sample text file for testing
└── README.md             # This file
```

## 🧪 Testing

Run the unit tests to verify functionality:

```bash
cd tests
python test_text_analyzer.py
```

All tests should pass, confirming that the core functionalities work correctly.

## 📊 Example Output

### Summary
```
Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
```

### Keywords
```
artificial, intelligence, machines, agents, environment, goals, learning, problem-solving
```

### Sentiment Analysis
```
TextBlob: Neutral (Polarity: 0.000, Subjectivity: 0.000)
VADER: Neutral (Compound: 0.000, Positive: 0.000, Neutral: 1.000, Negative: 0.000)
```

### Named Entities
```
AI (ORG)
```

### Text Statistics
```
Sentences: 3
Words: 67
Characters: 456
Characters (no spaces): 378
Avg words per sentence: 22.3
```

### Readability Analysis
```
Flesch Reading Ease: 45.2
Readability Level: Difficult
```

## 🔧 Technical Details

### Libraries Used

- **NLTK**: Natural Language Toolkit for text processing and summarization
- **spaCy**: Industrial-strength NLP for entity recognition and keyword extraction
- **TextBlob**: Simple API for diving into common NLP tasks
- **VADER Sentiment**: Valence Aware Dictionary and sEntiment Reasoner
- **Flask**: Web framework for the web interface

### Algorithms

- **Summarization**: Frequency-based extractive summarization using NLTK
- **Keyword Extraction**: Part-of-speech tagging and frequency analysis with spaCy
- **Sentiment Analysis**: Dual approach using TextBlob and VADER for comprehensive sentiment scoring
- **Entity Recognition**: Named Entity Recognition using spaCy's pre-trained models
- **Readability**: Flesch Reading Ease score calculation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NLTK team for the comprehensive natural language processing toolkit
- spaCy team for the industrial-strength NLP library
- TextBlob and VADER developers for sentiment analysis capabilities
- Flask team for the excellent web framework

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Port Already in Use**
   ```bash
   # Kill existing Flask processes
   pkill -f "python app.py"
   ```

### Performance Tips

- For large texts, consider increasing the summary sentence limit
- Use the `--no-entities` flag for faster processing if entity recognition is not needed
- The web interface is optimized for texts up to 10,000 characters

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Review the test cases in the `tests/` directory
3. Create an issue in the repository with detailed information about your problem

---

**Made with ❤️ using Python, NLTK, and spaCy**

