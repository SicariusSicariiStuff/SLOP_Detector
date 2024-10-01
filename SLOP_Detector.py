import sys
import os
import yaml
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer
import re

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_penalty_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def process_text(text, ignore_words, ignore_characters):
    for char in ignore_characters:
        text = text.replace(char, ' ')
    words = re.findall(r'\b\w+\b', text.lower())
    ignore_words_set = set(word.lower() for word in ignore_words)
    return [word for word in words if word not in ignore_words_set]

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
    for pattern in phrases:
        compiled_regex = re.compile(pattern, re.IGNORECASE)
        matches = compiled_regex.findall(text.lower())
        count = len(matches)
        if count > 0:
            phrase_counter[pattern] += count
    return phrase_counter

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
    elif 0.009 <= slop_score < 0.017:
        return 3
    elif 0.017 <= slop_score < 0.18:
        return 2
    else:
        return 1

def apply_penalties(text, penalties):
    penalty_score = 0.0
    for penalty_class in penalties.values():
        for penalty in penalty_class:
            count = text.lower().count(penalty['phrase'].lower())
            penalty_score += count * penalty['penalty']
    return penalty_score

def is_newline_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        return not first_line.endswith(',')

def analyze_file(filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases, ignore_words, ignore_characters, penalties):
    word_counter = Counter()
    total_words = 0
    gemma1_tokens = 0
    llama3_tokens = 0
    phrase_counter = Counter()
    total_penalty = 0.0

    print(f"Analyzing file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing lines"):
            text = line.strip()
            words = process_text(text, ignore_words, ignore_characters)
            word_counter.update(words)
            total_words += len(words)
            gemma1_tokens += count_tokens(text, tokenizer_GEMMA1)
            llama3_tokens += count_tokens(text, tokenizer_LLAMA3)
            phrase_counter.update(count_phrases(text, phrases))
            total_penalty += apply_penalties(text, penalties)

    sorted_phrase_counts = sorted(phrase_counter.items(), key=lambda x: x[1], reverse=True)
    total_gptisms = sum(phrase_counter.values())
    slop_score = (total_gptisms / total_words + total_penalty) if total_words > 0 else 0

    # Adjust the SLOP Coefficient based on the total word count
    if total_words > 10000 and slop_score > 0.001:
        adjusted_slop_score = adjust_slop_coefficient(slop_score, total_words)
        slop_score = adjusted_slop_score

    slop_rating = slop_to_score(slop_score)

    return word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts

def adjust_slop_coefficient(slop_score, total_words):
    # Define a reduction percentage (e.g., 10%)
    reduction_percentage = 0.25

    # Calculate the maximum reduction based on the total words
    max_reduction = slop_score * reduction_percentage

    # Calculate the number of word count chunks (e.g., every 10,000 words)
    word_count_chunks = total_words // 10000

    # Limit the number of chunks to avoid excessive reduction
    word_count_chunks = min(word_count_chunks, 5)  # Adjust this limit as needed

    # Apply reduction based on the number of chunks
    reduction = word_count_chunks * max_reduction / 5

    # Ensure the slop score doesn't become negative
    adjusted_slop_score = max(slop_score - reduction, 0.0)

    # Adjust the coefficient only if it's higher than 0.001
    if adjusted_slop_score > 0.001:
        adjusted_slop_score -= 0.0002

    return adjusted_slop_score

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
            output_file.write(f"  {phrase}: {format_large_numbers(count)}\n")
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

    if os.path.isfile(input_path):
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = base_filename + '_STATS'
    else:
        base_dir_name = os.path.basename(os.path.normpath(input_path))
        output_dir = base_dir_name + '_STATS'

    os.makedirs(output_dir, exist_ok=True)

    slop_data = load_yaml('SLOP.yml')
    phrases = slop_data['phrases']

    ignore_data = load_yaml('ignore.yml')
    ignore_words = set(ignore_data['ignore_words'])
    ignore_characters = set(ignore_data['ignore_characters'])

    penalty_data = load_penalty_yaml('penalty.yml')
    penalties = penalty_data['penalties']

    tokenizer_GEMMA1 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/2B_or_not_2B")
    tokenizer_LLAMA3 = AutoTokenizer.from_pretrained("SicariusSicariiStuff/LLAMA-3_8B_Unaligned")

    if os.path.isfile(input_path):
        word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(input_path, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases, ignore_words, ignore_characters, penalties)
        filename = os.path.splitext(os.path.basename(input_path))[0]
        export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                filepath = os.path.join(root, file)
                word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, phrases, ignore_words, ignore_characters, penalties)
                filename = os.path.splitext(file)[0]
                export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)

if __name__ == "__main__":
    main()
