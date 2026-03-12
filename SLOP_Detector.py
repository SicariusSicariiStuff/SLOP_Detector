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


# CHANGED: Supports two modes:
# 1) literal phrase (default) -> escaped + word boundaries
# 2) regex phrase when prefixed with "re:" -> compiled as-is
def build_phrase_regex(phrase: str):
    phrase = phrase.strip()

    # Raw regex mode from YAML:
    # Example: re:(?<!\w)ruin\s+(?:you|me|him|her|them)\s+for(?!\w)
    if phrase.startswith("re:"):
        raw_pattern = phrase[3:].strip()
        return re.compile(raw_pattern, re.IGNORECASE)

    # Literal mode: safe escaping + flexible spacing + anti-substring guards
    escaped = re.escape(phrase)
    escaped = escaped.replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


# CHANGED: Precompile phrases once before processing any files
def compile_phrases(phrases):
    compiled = []
    for phrase in phrases:
        phrase = phrase.strip()
        try:
            regex = build_phrase_regex(phrase)
        except re.error as e:
            raise ValueError(f"Bad regex in SLOP.yml ({phrase!r}): {e}") from e
        compiled.append({
            "source": phrase,
            "regex": regex,
            "is_regex": phrase.startswith("re:")
        })
    return compiled

# CHANGED: Precompile penalties once, also supports "re:" in penalty phrases


def compile_penalties(penalties):
    compiled = []
    for penalty_class in penalties.values():
        for penalty in penalty_class:
            p = penalty['phrase'].strip()
            try:
                regex = build_phrase_regex(p)
            except re.error as e:
                raise ValueError(
                    f"Bad regex in penalty.yml ({p!r}): {e}") from e
            compiled.append({
                'phrase': p,
                'penalty': penalty['penalty'],
                'regex': regex
            })
    return compiled


# CHANGED:
# - Uses compiled regexes instead of re.compile per call
# - For regex rules, reports the actual matched text in stats
# - Deduplicates matches by exact character span so the same text
#   fragment isn't counted twice when multiple rules overlap
def count_phrases(text, compiled_phrases):
    phrase_counter = Counter()
    used_spans = set()

    for rule in compiled_phrases:
        for m in rule["regex"].finditer(text):
            span = (m.start(), m.end())
            if span in used_spans:
                continue
            used_spans.add(span)

            if rule["is_regex"]:
                # Report actual matched text for regex rules
                hit = re.sub(r"\s+", " ", m.group(0).strip()).lower()
                phrase_counter[hit] += 1
            else:
                phrase_counter[rule["source"]] += 1

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


# CHANGED: Regex-based penalty matching - no more substring .count() false positives
# Deduplicates hits by span so overlapping rules don't stack on the same match
def apply_penalties(text, compiled_penalties):
    penalty_score = 0.0
    used_spans = set()

    for penalty in compiled_penalties:
        for m in penalty['regex'].finditer(text):
            span = (m.start(), m.end())
            if span in used_spans:
                continue
            used_spans.add(span)
            penalty_score += penalty['penalty']

    return penalty_score


def analyze_file(filepath, tokenizer_GEMMA1, tokenizer_LLAMA3, compiled_phrases, ignore_words, ignore_characters, compiled_penalties):
    word_counter = Counter()
    total_words = 0
    gemma1_tokens = 0
    llama3_tokens = 0
    phrase_counter = Counter()
    total_penalty = 0.0

    print(f"Analyzing file: {filepath}")

    # FIXED: pre-count lines so tqdm can show proper progress + ETA
    # (streaming the file iterator directly loses the length info)
    with open(filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # CHANGED: stream lines instead of readlines() to keep memory usage low
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing lines", total=total_lines):
            text = line.strip()
            words = process_text(text, ignore_words, ignore_characters)
            word_counter.update(words)
            total_words += len(words)
            gemma1_tokens += count_tokens(text, tokenizer_GEMMA1)
            llama3_tokens += count_tokens(text, tokenizer_LLAMA3)
            phrase_counter.update(count_phrases(text, compiled_phrases))
            total_penalty += apply_penalties(text, compiled_penalties)

    sorted_phrase_counts = sorted(
        phrase_counter.items(), key=lambda x: x[1], reverse=True)
    total_gptisms = sum(phrase_counter.values())
    slop_score = (total_gptisms / total_words +
                  total_penalty) if total_words > 0 else 0

    if total_words > 10000 and slop_score > 0.001:
        slop_score = adjust_slop_coefficient(slop_score, total_words)

    slop_rating = slop_to_score(slop_score)

    return word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts


def adjust_slop_coefficient(slop_score, total_words):
    reduction_percentage = 0.25
    max_reduction = slop_score * reduction_percentage
    word_count_chunks = min(total_words // 10000, 5)
    reduction = word_count_chunks * max_reduction / 5
    adjusted_slop_score = max(slop_score - reduction, 0.0)
    if adjusted_slop_score > 0.001:
        adjusted_slop_score -= 0.0002
    return adjusted_slop_score


def export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts):
    output_filepath = os.path.join(output_dir, filename + '_Statistics.txt')

    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        output_file.write(f"SLOP Score: {slop_rating}\n")
        output_file.write(
            f"Total Words: {format_large_numbers(total_words)}\n")
        output_file.write(
            f"GEMMA1 Tokens: {format_large_numbers(gemma1_tokens)}\n")
        output_file.write(
            f"LLAMA3 Tokens: {format_large_numbers(llama3_tokens)}\n")
        output_file.write(
            f"Total GPT-isms: {format_large_numbers(total_gptisms)}\n")
        output_file.write(f"SLOP Coefficient: {slop_score:.6f}\n")
        output_file.write("\nGPT-ism found:\n")
        for phrase, count in sorted_phrase_counts:
            output_file.write(f"  {phrase}: {format_large_numbers(count)}\n")
        output_file.write("=" * 40 + "\n")

        for word, count in word_counter.most_common():
            percentage = (count / total_words) * \
                100 if total_words > 0 else 0.0
            output_file.write(
                f"{word:<20}{format_large_numbers(count):<10}{percentage:.2f}%\n")

    print(f"Statistics exported to {output_filepath}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python SLOP_Detector.py <file_or_directory>")
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

    # CHANGED: compile once, pass compiled objects into analyze_file
    compiled_phrases = compile_phrases(phrases)
    compiled_penalties = compile_penalties(penalties)

    tokenizer_GEMMA1 = AutoTokenizer.from_pretrained(
        "SicariusSicariiStuff/2B_or_not_2B")
    tokenizer_LLAMA3 = AutoTokenizer.from_pretrained(
        "SicariusSicariiStuff/LLAMA-3_8B_Unaligned")

    if os.path.isfile(input_path):
        word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(
            input_path,
            tokenizer_GEMMA1,
            tokenizer_LLAMA3,
            compiled_phrases,
            ignore_words,
            ignore_characters,
            compiled_penalties
        )
        filename = os.path.splitext(os.path.basename(input_path))[0]
        export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens,
                          llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                filepath = os.path.join(root, file)
                word_counter, total_words, gemma1_tokens, llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts = analyze_file(
                    filepath,
                    tokenizer_GEMMA1,
                    tokenizer_LLAMA3,
                    compiled_phrases,
                    ignore_words,
                    ignore_characters,
                    compiled_penalties
                )
                filename = os.path.splitext(file)[0]
                export_statistics(output_dir, filename, word_counter, total_words, gemma1_tokens,
                                  llama3_tokens, total_gptisms, slop_score, slop_rating, sorted_phrase_counts)


if __name__ == "__main__":
    main()
