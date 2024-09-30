import sys
import os
import json
import yaml
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

def process_text(text):
    words = text.split()
    return [word.lower().strip('.,!?;"\'') for word in words]

def format_large_numbers(number):
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return str(number)

def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def count_phrases(text, phrases):
    phrase_counter = Counter()
    for phrase in phrases:
        count = text.lower().count(phrase.lower())
        if count > 0:
            phrase_counter[phrase] += count
    return phrase_counter

def json_slop_detection(json_data):
    gpt_texts = []
    if isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and 'conversations' in item:
                for conv in item['conversations']:
                    if isinstance(conv, dict) and conv.get('from') == 'gpt':
                        gpt_value = conv.get('value')
                        if gpt_value:
                            gpt_texts.append(gpt_value)
    return ' '.join(gpt_texts)

def slop_to_score(slop_score):
    if slop_score < 0.0009:
        return 10
    elif 0.0009 <= slop_score < 0.0016:
        return 9
    elif 0.0016 <= slop_score < 0.002:
        return 8
    elif 0.002 <= slop_score < 0.0027:
        return 7
    elif 0.0027 <= slop_score < 0.003:
        return 6
    elif 0.003 <= slop_score < 0.005:
        return 5
    elif 0.005 <= slop_score < 0.009:
        return 4
    elif 0.009 <= slop_score < 0.014:
        return 3
    elif 0.014 <= slop_score < 0.02:
        return 2
    else:
        return 1

def analyze_file(filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases):
    word_counter = Counter()
    total_words = 0
    gemma1_tokens = 0
    llama3_tokens = 0
    phrase_counter = Counter()

    file_ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, 'r', encoding='utf-8') as file:
        if file_ext in ['.json', '.jsonl']:
            # Process JSON file
            json_data = json.load(file)
            text = json_slop_detection(json_data)
            words = process_text(text)
            word_counter.update(words)
            total_words += len(words)
            gemma1_tokens += count_tokens(text, tokenizer_GEMMA1)
            llama3_tokens += count_tokens(text, tokenizer_LLAMA3)
            phrase_counter.update(count_phrases(text, phrases))
        else:
            # Process TXT file
            lines = file.readlines()
            for line in tqdm(lines, total=len(lines), desc="Processing TXT lines"):
                text = line.strip()
                words = process_text(text)
                word_counter.update(words)
                total_words += len(words)
                gemma1_tokens += count_tokens(text, tokenizer_GEMMA1)
                llama3_tokens += count_tokens(text, tokenizer_LLAMA3)
                phrase_counter.update(count_phrases(text, phrases))

    # Sort phrases by count in descending order
    sorted_phrase_counts = sorted(phrase_counter.items(), key=lambda x: x[1], reverse=True)

    # Calculate total GPT-isms
    total_gptisms = sum(phrase_counter.values())

    # Calculate SLOP Score
    slop_score = total_gptisms / total_words if total_words > 0 else 0

    # Determine SLOP rating
    slop_rating = slop_to_score(slop_score)

    return word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts

def export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts):
    output_filepath = os.path.join(output_dir, filename + '_Statistics.txt')

    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        output_file.write(f"SLOP Score: {slop_rating}\n")
        output_file.write(f"Total Words: {format_large_numbers(total_words)}\n")
        output_file.write(f"GEMMA1 Tokens: {format_large_numbers(gemma1_tokens)}\n")
        output_file.write(f"LLAMA3 Tokens: {format_large_numbers(llama3_tokens)}\n")
        output_file.write(f"Total GPT-isms: {format_large_numbers(total_gptisms)}\n")
        output_file.write(f"SLOP Coefficient: {slop_score:.6f}\n")
        output_file.write("\nGPT-ism found:\n")
        for phrase, count in sorted_phrase_counts:
            output_file.write(f"  {phrase}: {format_large_numbers(count)}\n")
        output_file.write("="*40 + "\n")

        for word, count in word_counter.most_common():
            percentage = (count / total_words) * 100
            output_file.write(f"{word:<20}{format_large_numbers(count):<10}{percentage:.2f}%\n")

    print(f"Statistics exported to {output_filepath}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_text.py <file_or_directory>")
        return

    input_path = sys.argv[1]

    # Determine output directory name
    if os.path.isdir(input_path):
        base_dir_name = os.path.basename(os.path.normpath(input_path))
        output_dir = base_dir_name + '_STATS'
    else:
        base_dir_name = os.path.basename(os.path.dirname(input_path))
        output_dir = base_dir_name + '_STATS'

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load phrases from YAML file
    with open('SLOP.yml', 'r', encoding='utf-8') as yaml_file:
        phrases = yaml.safe_load(yaml_file)['phrases']

    # Initialize tokenizers
    tokenizer_GEMMA1 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/2B_or_not_2B")
    tokenizer_LLAMA3 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/LLAMA-3_8B_Unaligned")

    # Process files
    if os.path.isfile(input_path):
        word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(input_path, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases)
        filename = os.path.splitext(os.path.basename(input_path))[0]
        export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                filepath = os.path.join(root, file)
                word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases)
                filename = os.path.splitext(file)[0]
                export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)

if __name__ == "__main__":
    main()
