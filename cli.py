#!/usr/bin/env python3
"""
Command-line interface for the AI Text Analyzer
"""

import argparse
import sys
import json
from text_analyzer import TextAnalyzer

def main():
    parser = argparse.ArgumentParser(description='AI Text Analyzer - Powerful text analysis with AI-driven insights')
    parser.add_argument('input', help='Input text file path or text string')
    parser.add_argument('-f', '--file', action='store_true', help='Treat input as file path (default: treat as text)')
    parser.add_argument('-s', '--summary', type=int, default=3, help='Number of sentences in summary (default: 3)')
    parser.add_argument('-k', '--keywords', type=int, default=10, help='Number of keywords to extract (default: 10)')
    parser.add_argument('-o', '--output', help='Output file path (JSON format)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--no-summary', action='store_true', help='Skip text summarization')
    parser.add_argument('--no-sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--no-entities', action='store_true', help='Skip entity extraction')
    parser.add_argument('--no-stats', action='store_true', help='Skip text statistics')
    parser.add_argument('--no-readability', action='store_true', help='Skip readability analysis')
    parser.add_argument('--word-cloud', action='store_true', help='Include word frequency data for word clouds')
    parser.add_argument('--learning-summary', action='store_true', help='Generate learning-focused summary for better understanding')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = TextAnalyzer()
        
        # Get text content
        if args.file:
            try:
                text = analyzer.read_text_file(args.input)
                print(f"📁 Analyzing file: {args.input}")
            except FileNotFoundError:
                print(f"❌ Error: File '{args.input}' not found.", file=sys.stderr)
                sys.exit(1)
        else:
            text = args.input
            print("📝 Analyzing provided text...")
        
        if not text.strip():
            print("❌ Error: No text content to analyze.", file=sys.stderr)
            sys.exit(1)
        
        # Perform analysis based on options
        results = {}
        
        if not args.no_summary:
            print("🔍 Generating summary...")
            results['summary'] = analyzer.summarize_text_nltk(text, args.summary)
        
        print("🎯 Extracting keywords...")
        results['keywords'] = analyzer.extract_keywords_spacy(text, args.keywords)
        
        if not args.no_sentiment:
            print("😊 Analyzing sentiment...")
            results['sentiment_textblob'] = analyzer.analyze_sentiment_textblob(text)
            results['sentiment_vader'] = analyzer.analyze_sentiment_vader(text)
        
        if not args.no_entities:
            print("🏷️ Extracting entities...")
            results['entities'] = analyzer.extract_entities(text)
        
        if not args.no_stats:
            print("📊 Calculating statistics...")
            results['statistics'] = analyzer.get_text_statistics(text)
        
        if not args.no_readability:
            print("📖 Analyzing readability...")
            results['readability'] = analyzer.analyze_readability(text)
        
        if args.word_cloud:
            print("☁️ Generating word cloud data...")
            results['word_cloud'] = analyzer.get_word_frequency_for_cloud(text)
        
        if args.learning_summary:
            print("📚 Generating learning summary...")
            results['learning_summary'] = analyzer.create_learning_summary(text)
        
        # Output results
        if args.json or args.output:
            output_data = results
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to: {args.output}")
            else:
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
        else:
            # Pretty print results
            print("\n" + "="*60)
            print("🤖 AI TEXT ANALYSIS RESULTS")
            print("="*60)
            
            if 'summary' in results:
                print(f"\n📝 SUMMARY:")
                print("-" * 40)
                print(results['summary'])
            
            if 'learning_summary' in results:
                print(f"\n📚 LEARNING SUMMARY:")
                print("-" * 40)
                print(results['learning_summary'])
            
            print(f"\n🎯 KEYWORDS:")
            print("-" * 40)
            print(", ".join(results['keywords']))
            
            if 'sentiment_textblob' in results:
                print(f"\n😊 SENTIMENT ANALYSIS (TextBlob):")
                print("-" * 40)
                sentiment = results['sentiment_textblob']
                print(f"Sentiment: {sentiment['sentiment']}")
                print(f"Polarity: {sentiment['polarity']:.3f}")
                print(f"Subjectivity: {sentiment['subjectivity']:.3f}")
                
                print(f"\n🎭 SENTIMENT ANALYSIS (VADER):")
                print("-" * 40)
                vader = results['sentiment_vader']
                print(f"Sentiment: {vader['sentiment']}")
                print(f"Compound: {vader['compound']:.3f}")
                print(f"Positive: {vader['positive']:.3f}")
                print(f"Neutral: {vader['neutral']:.3f}")
                print(f"Negative: {vader['negative']:.3f}")
            
            if 'entities' in results and results['entities']:
                print(f"\n🏷️ NAMED ENTITIES:")
                print("-" * 40)
                for entity, label in results['entities']:
                    print(f"{entity} ({label})")
            
            if 'statistics' in results:
                print(f"\n📊 TEXT STATISTICS:")
                print("-" * 40)
                stats = results['statistics']
                print(f"Sentences: {stats['sentences']}")
                print(f"Words: {stats['words']}")
                print(f"Characters: {stats['characters']}")
                print(f"Characters (no spaces): {stats['characters_no_spaces']}")
                print(f"Avg words per sentence: {stats['avg_words_per_sentence']:.1f}")
            
            if 'readability' in results:
                print(f"\n📖 READABILITY ANALYSIS:")
                print("-" * 40)
                readability = results['readability']
                print(f"Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
                print(f"Readability Level: {readability['readability_level']}")
            
            if 'word_cloud' in results:
                print(f"\n☁️ WORD CLOUD DATA:")
                print("-" * 40)
                word_cloud = results['word_cloud']
                for word, freq in list(word_cloud.items())[:10]:  # Show top 10
                    print(f"{word}: {freq}")
        
        print("\n✅ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

