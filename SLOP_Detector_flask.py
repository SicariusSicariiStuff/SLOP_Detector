from flask import Flask, request, render_template, send_file, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from SLOP_Detector import (
    load_yaml, load_penalty_yaml, analyze_file, export_statistics,
    AutoTokenizer
)

app = Flask(__name__)

# Load necessary data and models
try:
    slop_data = load_yaml('SLOP.yml')
    phrases = slop_data['phrases']

    ignore_data = load_yaml('ignore.yml')
    ignore_words = set(ignore_data['ignore_words'])
    ignore_characters = set(ignore_data['ignore_characters'])

    penalty_data = load_penalty_yaml('penalty.yml')
    penalties = penalty_data['penalties']

    tokenizer_GEMMA1 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/2B_or_not_2B")
    tokenizer_LLAMA3 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/LLAMA-3_8B_Unaligned")
except Exception as e:
    print(f"Error loading necessary data and models: {e}")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    filepath = os.path.join(tmpdir, filename)
                    file.save(filepath)
                    
                    # Analyze the file
                    word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(
                        filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases, ignore_words, ignore_characters, penalties
                    )
                    
                    # Export statistics
                    output_dir = tempfile.mkdtemp()
                    stats_filename = os.path.splitext(filename)[0]
                    export_statistics(
                        output_dir, stats_filename, word_counter, total_words,
                        gemma1_tokens, llama3_tokens, total_gptisms, slop_score,
                        slop_rating, sorted_phrase_counts
                    )
                    
                    # Send the statistics file
                    stats_file = os.path.join(output_dir, f'{stats_filename}_Statistics.txt')
                    return send_file(stats_file, as_attachment=True, download_name=f'{stats_filename}_Statistics.txt')
            except Exception as e:
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return render_template('upload.html')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5109, debug=True)
