import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def read_text_file(filepath):
    """
    Reads the content of a text file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def summarize_text_nltk(text, num_sentences=3):
    """
    Summarizes text using NLTK by extracting the most important sentences.
    """
    sentences = sent_tokenize(text)
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
        return ""
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
    
    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:  # Avoid very long sentences
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    
    # Sort sentences by score and return the top ones
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ' '.join(summarized_sentences)

def extract_keywords_spacy(text):
    """
    Extracts keywords from text using spaCy.
    """
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
    # Simple frequency-based keyword selection
    keyword_freq = nltk.FreqDist(keywords)
    return [word for word, freq in keyword_freq.most_common(5)]

if __name__ == "__main__":
    # Example usage (for testing purposes)
    dummy_file_path = "/home/ubuntu/text_analyzer/dummy_text.txt"
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        f.write("Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem-solving'.")

    try:
        text_content = read_text_file(dummy_file_path)
        print("\n--- Original Text ---")
        print(text_content)

        summary = summarize_text_nltk(text_content, num_sentences=2)
        print("\n--- Summary (NLTK) ---")
        print(summary)

        keywords = extract_keywords_spacy(text_content)
        print("\n--- Keywords (spaCy) ---")
        print(keywords)

    except FileNotFoundError as e:
        print(e)


