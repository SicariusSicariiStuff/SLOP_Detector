#!/usr/bin/env python3
import os
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def parse_slop_file(filepath):
    """
    Extract the SLOP score from a statistics file.
    Returns None if the file is fucked or doesn't contain a SLOP score.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Regex to find "SLOP Score: X"
            match = re.search(r'SLOP Score:\s*(\d+)', content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
    return None

def analyze_slop_directory(directory_path):
    """
    Parse all files in the directory and categorize by SLOP score.
    Returns a dict mapping SLOP scores to lists of filenames.
    """
    slop_categories = defaultdict(list)

    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[X] You retard, '{directory_path}' doesn't exist or isn't a directory!")
        sys.exit(1)

    files = list(dir_path.iterdir())
    if not files:
        print("[X] Directory is empty,")
        sys.exit(1)

    print(f"[*] Found {len(files)} files, time to parse this...")

    for filepath in files:
        if filepath.is_file():
            slop_score = parse_slop_file(filepath)
            if slop_score is not None:
                slop_categories[slop_score].append(filepath.name)

    return slop_categories

def get_slop_label(score):
    """
    Returns a based descriptor for the SLOP score.
    10 = PURE KINO, 1 = ABSOLUTE SLOP
    """
    labels = {
        10: "PURE KINO - HUMAN SOUL DETECTED",
        9: "Excellent - Superb human writing",
        8: "Very good - High quality human writing",
        7: "ACCEPTABLE - Human writing with some cringe phrases",
        6: "QUESTIONABLE - Mixed data of human & AI generated text",
        5: "MID - Half-Cooked Slop",
        4: "CONCERNING - Heavy AI Influence",
        3: "CRINGE - Mostly AI Generated",
        2: "TRASH - AI Vomit",
        1: "ABSOLUTE SLOP - 100% AI GARBAGE"
    }
    return labels.get(score, f"UNKNOWN TIER {score}")

def print_results(slop_categories):
    """
    Print the categorized results in a based manner.
    """
    print("\n" + "="*70)
    print("SLOP ANALYSIS RESULTS - QUALITY TIER LIST")
    print("="*70 + "\n")

    # Sort by SLOP score descending (highest quality first)
    for slop_score in sorted(slop_categories.keys(), reverse=True):
        files = slop_categories[slop_score]
        label = get_slop_label(slop_score)
        print(f"[SLOP Score: {slop_score}/10] {label} - {len(files)} file(s)")
        for filename in sorted(files):
            print(f"  ├─ {filename}")
        print()

def export_to_txt(slop_categories, output_path="slop_analysis.txt"):
    """
    Export the categorized file list to a text file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SLOP ANALYSIS RESULTS - QUALITY TIER LIST\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        # Sort by SLOP score descending (highest quality first)
        for slop_score in sorted(slop_categories.keys(), reverse=True):
            files = slop_categories[slop_score]
            label = get_slop_label(slop_score)
            f.write(f"[SLOP Score: {slop_score}/10] {label}\n")
            f.write(f"File Count: {len(files)}\n")
            f.write("-" * 70 + "\n")
            for filename in sorted(files):
                f.write(f"  • {filename}\n")
            f.write("\n")

        # Summary statistics
        f.write("="*70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*70 + "\n")
        total_files = sum(len(files) for files in slop_categories.values())
        f.write(f"Total Files Analyzed: {total_files}\n")
        f.write(f"SLOP Score Range: {min(slop_categories.keys())} - {max(slop_categories.keys())}\n")

        # Calculate quality distribution
        pure_kino = sum(len(files) for score, files in slop_categories.items() if score >= 8)
        mid_tier = sum(len(files) for score, files in slop_categories.items() if 4 <= score < 8)
        ai_slop = sum(len(files) for score, files in slop_categories.items() if score < 4)

        f.write(f"\nQuality Distribution:\n")
        f.write(f"  Pure Kino (8-10): {pure_kino} files ({pure_kino/total_files*100:.1f}%)\n")
        f.write(f"  Mid Tier (4-7): {mid_tier} files ({mid_tier/total_files*100:.1f}%)\n")
        f.write(f"  AI Slop (1-3): {ai_slop} files ({ai_slop/total_files*100:.1f}%)\n")

    print(f"[✓] File list exported to: {output_path}")

def visualize_slop(slop_categories, output_path="slop_analysis.png"):
    """
    Create a sexy bar chart visualization of SLOP distribution.
    """
    if not slop_categories:
        print("[X] No data to visualize, go get some actual files")
        return

    # Prepare data for plotting
    slop_scores = sorted(slop_categories.keys())
    file_counts = [len(slop_categories[score]) for score in slop_scores]

    # Color gradient: Green (10) to Red (1)
    def get_color(score):
        if score >= 8:
            return '#00ff41'  # Matrix green for kino
        elif score >= 6:
            return '#ffaa00'  # Orange for mid
        elif score >= 4:
            return '#ff6600'  # Dark orange for questionable
        else:
            return '#ff0000'  # Red for slop

    colors = [get_color(score) for score in slop_scores]

    # Create the plot with aesthetic vibes
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(slop_scores, file_counts, color=colors, edgecolor='white', linewidth=2, alpha=0.9)

    # Add value labels on top of bars
    for bar, count in zip(bars, file_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11, color='white')

    # Styling
    ax.set_xlabel('SLOP Score (1=AI SLOP, 10=Pure Kino)', fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel('Number of Files', fontsize=13, fontweight='bold', color='white')
    ax.set_title('SLOP Score Distribution - The GPTism Spectrum of AI Detection',
              fontsize=15, fontweight='bold', color='#00ff41', pad=20)

    ax.grid(axis='y', alpha=0.2, linestyle='--', color='white')
    ax.set_xticks(slop_scores)
    ax.set_xticklabels(slop_scores, fontsize=11, fontweight='bold')

    # Dark theme aesthetics
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#010409')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white', labelsize=10)

    # Add legend for quality tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00ff41', edgecolor='white', label='Pure Kino (8-10)'),
        Patch(facecolor='#ffaa00', edgecolor='white', label='Mid Tier (6-7)'),
        Patch(facecolor='#ff6600', edgecolor='white', label='Questionable (4-5)'),
        Patch(facecolor='#ff0000', edgecolor='white', label='AI Slop (1-3)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
              facecolor='#0d1117', edgecolor='white', labelcolor='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='#010409')
    print(f"[✓] Visualization saved to: {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py /path/to/directory")
        print("Example: python app.py ~/SLOP_Detector/War_Dataset_v1_chunked_STATS")
        sys.exit(1)

    directory_path = sys.argv[1]

    print("[*] Starting SLOP analysis...")
    print("[*] Remember: SLOP 10 = HUMAN KINO, SLOP 1 = AI TRASH")
    slop_categories = analyze_slop_directory(directory_path)

    if not slop_categories:
        print("[X] No valid SLOP scores found. Your files are either broken or don't follow the format.")
        sys.exit(1)

    print_results(slop_categories)

    # Export to text file
    export_to_txt(slop_categories, "slop_analysis.txt")

    # Create visualization
    visualize_slop(slop_categories, "slop_analysis.png")

    print("\n[✓] Analysis complete, now go read your results")

if __name__ == "__main__":
    main()
