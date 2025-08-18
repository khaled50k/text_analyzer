from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from text_analyzer import TextAnalyzer
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Initialize the text analyzer
analyzer = TextAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and file.filename.endswith('.txt'):
        try:
            # Read file content
            content = file.read().decode('utf-8')
            
            # Perform comprehensive analysis
            analysis = analyzer.comprehensive_analysis(content)
            
            return render_template('results.html', 
                                 original_text=content,
                                 analysis=analysis,
                                 filename=file.filename)
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Please upload a .txt file')
        return redirect(url_for('index'))

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        text = request.form.get('text', '')
        if not text.strip():
            flash('Please enter some text to analyze')
            return redirect(url_for('index'))
        
        # Perform comprehensive analysis
        analysis = analyzer.comprehensive_analysis(text)
        
        return render_template('results.html', 
                             original_text=text,
                             analysis=analysis,
                             filename="Direct Input")
    except Exception as e:
        flash(f'Error analyzing text: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for text analysis"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        analysis = analyzer.comprehensive_analysis(text)
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

