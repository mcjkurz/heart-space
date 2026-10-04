#!/usr/bin/env python3
"""
Chinese text normalization script.
Normalizes Chinese text by:
1. Converting traditional Chinese to simplified Chinese using OpenCC
2. Filtering out lines containing banned words
3. Removing chapter indicators (第X章, 第X回, etc.)
4. Removing lines starting with circled numbers (①, ②, ③, etc.) or reference marks (※)
5. Removing empty lines and all whitespace within lines
6. Removing all numeric characters
7. Normalizing punctuation to full-width Chinese characters
8. Filtering out lines shorter than minimum length (default: 2 chars)
9. Normalizing output filenames (traditional to simplified Chinese)

Usage (run from project root):
  python scripts/normalize_texts.py                           # process all default periods
  python scripts/normalize_texts.py data/texts/mingqing --output-dir data/texts_normalized/mingqing
  python scripts/normalize_texts.py input.txt --output-dir custom_output/
  python scripts/normalize_texts.py input.txt --min-length 3

Output structure:
- No arguments: processes all period folders under data/texts/ into data/texts_normalized/
- For single file: creates output file with same name in specified directory or filename_simplified.txt
- For folders: creates folder with _simplified suffix or specified output directory
"""

import sys
import os
import argparse
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import opencc
import re


class ChineseNormalizer:
    """Chinese text normalizer using OpenCC and punctuation normalization"""
    
    def __init__(self, min_length_line=3):
        """Initialize the normalizer with OpenCC converter"""
        # Initialize OpenCC converter for traditional to simplified Chinese
        self.converter = opencc.OpenCC('t2s')  # traditional to simplified
        
        # Unicode-based mapping for converting ASCII to full-width characters
        self.half2full = dict((i, i + 0xFEE0) for i in range(0x21, 0x7F))
        self.half2full[0x20] = 0x3000  # space to ideographic space
        
        # Additional specific character mappings
        self.specific_mappings = {
            '「': '"',
            '」': '"',
        }
        
        # Set of banned words to filter out lines containing them
        self.banned_words = set(["txt", "book", "com", "net", "org", "www"])
        
        # Minimum line length threshold
        self.min_length_line = min_length_line
        
        # Regex pattern for chapter indicators (第 + numbers + chapter/episode markers)
        # Matches: 第X章, 第X回, 第X节, 第X卷, 第X部, 第X篇, etc.
        # Numbers can be Chinese numerals (一二三四五六七八九十百千万零〇) or Arabic digits
        self.chapter_pattern = re.compile(
            r'第[一二三四五六七八九十百千万零〇\d]+[章回节卷部篇]'
        )
        
    def convert_to_simplified(self, text):
        """Convert traditional Chinese to simplified Chinese"""
        return self.converter.convert(text)
        
    def remove_numbers(self, text):
        """Remove all numeric characters from the text"""
        return re.sub(r'\d', '', text)
        
    def normalize_punctuation(self, text):
        """Convert ASCII characters to full-width and apply specific mappings"""
        # First apply Unicode-based ASCII to full-width conversion
        text = text.translate(self.half2full)
        
        # Then apply specific character mappings
        for old_char, new_char in self.specific_mappings.items():
            text = text.replace(old_char, new_char)
        
        return text
        
    def remove_whitespace_and_empty_lines(self, text):
        """Remove empty lines and all whitespace within lines"""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Remove all whitespace within the line
            line_no_whitespace = re.sub(r'\s', '', line)
            # Only keep non-empty lines
            if line_no_whitespace.strip():
                processed_lines.append(line_no_whitespace.strip())
        
        return '\n'.join(processed_lines)
        
    def filter_banned_words(self, text):
        """Remove lines that contain any banned words (case-insensitive regex matching)"""
        if not self.banned_words:
            return text
            
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Check if line contains any banned words using case-insensitive regex
            contains_banned = any(re.search(banned_word, line, re.IGNORECASE) for banned_word in self.banned_words)
            if not contains_banned:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def remove_chapter_indicators(self, text):
        """Remove chapter indicators (第X章, 第X回, etc.) from text line by line"""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Remove all chapter indicator patterns from the line
            cleaned_line = self.chapter_pattern.sub('', line)
            processed_lines.append(cleaned_line)
        
        return '\n'.join(processed_lines)
    
    def remove_circled_number_lines(self, text):
        """Remove lines that start with circled numbers (①②③ etc.) or reference marks (※)"""
        lines = text.split('\n')
        filtered_lines = []
        
        # Circled numbers (①-⑳, ㉑-㊿) and reference mark (※)
        marker_pattern = re.compile(r'^[\u2460-\u2473\u3251-\u325F\u32B1-\u32BF※]')
        
        for line in lines:
            if not marker_pattern.match(line):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def filter_short_lines(self, text):
        """Remove lines shorter than min_length_line characters"""
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            if len(line) >= self.min_length_line:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
        
    def normalize_filename(self, filename):
        """Normalize a filename by converting traditional Chinese characters to simplified"""
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        simplified_stem = self.converter.convert(stem)
        return f"{simplified_stem}{suffix}"

    def normalize_text(self, text):
        """Apply all normalization steps to the text"""
        # Step 1: Convert to simplified Chinese
        text = self.convert_to_simplified(text)

        # Step 2: Filter out lines containing banned words
        text = self.filter_banned_words(text)

        # Step 3: Remove chapter indicators (第X章, 第X回, etc.)
        text = self.remove_chapter_indicators(text)

        # Step 4: Remove lines starting with circled numbers (①②③) or reference marks (※)
        text = self.remove_circled_number_lines(text)

        # Step 5: Remove whitespace and empty lines
        text = self.remove_whitespace_and_empty_lines(text)
        
        # Step 6: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 7: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 8: Filter out lines shorter than minimum length
        text = self.filter_short_lines(text)
        
        return text
        
    def normalize_file(self, input_path, output_path=None):
        """Normalize a single text file"""
        input_path_obj = Path(input_path)
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        #print(f"Read {len(text)} characters from {input_path}")
        
        # Normalize the text
        normalized_text = self.normalize_text(text)
        
        # Determine output path
        normalized_name = self.normalize_filename(input_path_obj.name)
        if output_path is None:
            stem = Path(normalized_name).stem
            suffix = Path(normalized_name).suffix
            output_path = input_path_obj.parent / f"{stem}_simplified{suffix}"
        else:
            output_path = Path(output_path)
            if output_path.is_dir() or not output_path.suffix:
                output_path = output_path / normalized_name
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save normalized text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(normalized_text)
        
        #print(f"Normalized text saved to {output_path}")
        #print(f"Original length: {len(text)} characters")
        #print(f"Normalized length: {len(normalized_text)} characters")
        
        return output_path
        
    def normalize_folder(self, input_folder, output_dir=None):
        """Normalize all .txt files in a folder"""
        input_path = Path(input_folder)
        if not input_path.is_dir():
            raise ValueError(f"'{input_folder}' is not a directory")
            
        # Determine output folder
        if output_dir is None:
            output_folder = Path(f"{input_path}_simplified")
        else:
            output_folder = Path(output_dir)
        
        # Create output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all .txt files in the input folder
        txt_files = list(input_path.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {input_folder}")
            return
            
        print(f"Found {len(txt_files)} .txt files in {input_folder}")
        print(f"Output directory: {output_folder}")
        
        # Process each file
        for txt_file in tqdm(txt_files, desc="Processing files"):
            #print(f"\nProcessing {txt_file.name}...")
            
            # Read input file
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            #print(f"Read {len(text)} characters from {txt_file.name}")
            
            # Normalize the text
            normalized_text = self.normalize_text(text)
            
            # Save to output folder with normalized filename
            output_file = output_folder / self.normalize_filename(txt_file.name)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(normalized_text)
            
            #print(f"Normalized text saved to {output_file}")
            #print(f"Original length: {len(text)} → Normalized length: {len(normalized_text)} characters")
        
        print(f"\nAll files processed. Results saved in {output_folder}")
        return output_folder


DEFAULT_INPUT_DIR = "data/texts"
DEFAULT_OUTPUT_DIR = "data/texts_normalized"
PERIODS = ["mingqing", "late_qing", "republican", "socialist", "contemporary"]


def main():
    parser = argparse.ArgumentParser(description='Chinese text normalization script')
    parser.add_argument('input_path', nargs='?', default=None,
                        help='Input text file or folder to normalize (omit to process all periods)')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir',
                        help='Output directory (default: input_name_simplified)')
    parser.add_argument('--min-length', '--min_length', dest='min_length', type=int, default=3,
                        help='Minimum line length to keep (default: 3)')
    
    args = parser.parse_args()
    normalizer = ChineseNormalizer(min_length_line=args.min_length)

    if args.input_path is None:
        input_dir = DEFAULT_INPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR
        print(f"No input path given — processing all periods under {input_dir}/")
        
        # Clean existing output folders before processing
        for period in PERIODS:
            period_output = os.path.join(output_dir, period)
            if os.path.isdir(period_output):
                print(f"Cleaning existing folder: {period_output}")
                shutil.rmtree(period_output)
        
        for period in PERIODS:
            period_input = os.path.join(input_dir, period)
            period_output = os.path.join(output_dir, period)
            if not os.path.isdir(period_input):
                print(f"  Skipping {period}: {period_input} not found")
                continue
            print(f"\n=== {period} ===")
            normalizer.normalize_folder(period_input, period_output)
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' not found.")
        sys.exit(1)
    
    if os.path.isfile(input_path):
        normalizer.normalize_file(input_path, args.output_dir)
    elif os.path.isdir(input_path):
        normalizer.normalize_folder(input_path, args.output_dir)
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
