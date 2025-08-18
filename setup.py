#!/usr/bin/env python3
"""
Setup script for AI Text Analyzer
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

def download_spacy_model():
    """Download spaCy language model"""
    print("🧠 Downloading spaCy language model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def main():
    """Main setup function"""
    print("🚀 Setting up AI Text Analyzer...")
    
    try:
        install_requirements()
        download_nltk_data()
        download_spacy_model()
        
        print("\n✅ Setup completed successfully!")
        print("\n🎉 You can now use the AI Text Analyzer:")
        print("   • Web interface: python app.py")
        print("   • Command line: python cli.py --help")
        print("   • Python module: from text_analyzer import TextAnalyzer")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

