#!/usr/bin/env python3
"""
Pronoun resolution for the novel 蹉跎岁月 (Wasted Years).

Replaces pronouns 他/她 with character names using an LLM API.
Requires a POE_API_KEY environment variable (set in .env or shell).

Usage (run from project root):
  python experiments/suiyue.py
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import openai
from tqdm import tqdm

api_key = os.environ.get("POE_API_KEY")
if not api_key:
    print("Error: POE_API_KEY environment variable is not set.")
    print("Create a .env file in the project root with:")
    print("  POE_API_KEY=your_key_here")
    sys.exit(1)

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.poe.com/v1",
)

characters = {
    "柯碧舟": ["柯碧舟", "小柯", "碧舟"],
    "杜见春": ["杜见春", "见春"],
    "苏道诚": ["苏道诚", "小苏"],
    "王连发": ["王连发", "小王"],
    "肖永川": ["肖永川"],
    "唐惠娟": ["唐惠娟", "小唐", "惠娟"],
    "华雯雯": ["华雯雯", "雯雯", "华姑娘"],
    "邵大山": ["邵大山", "大山伯", "大山哥", "幺公"],
    "邵玉蓉": ["邵玉蓉", "玉蓉"],
    "左定法": ["左定法", "左主任"]
}

model = "GPT-5-mini"

character_names = list(characters.keys())
system_prompt = f"""You are a text processing assistant performing pronoun reference resolution. Your ONLY task is to replace the pronouns 他 and 她 with character names.

Target characters (ONLY replace pronouns referring to these characters):
{', '.join(character_names)}

EXAMPLE:
Input: "柯碧舟的脸涨得绯红绯红，为了掩饰自己的忐忑不安，他伸手拿过几根干柴，支支吾吾地说："
Output: "柯碧舟的脸涨得绯红绯红，为了掩饰自己的忐忑不安，柯碧舟伸手拿过几根干柴，支支吾吾地说："

CRITICAL RULES:
1. REPLACE 他 → character's FULL name (e.g., 柯碧舟, 杜见春, etc.)
2. REPLACE 她 → character's FULL name
3. ONLY replace if the pronoun clearly refers to one of the characters listed above
4. If ambiguous or refers to someone else, keep the original pronoun
5. Preserve ALL other text EXACTLY - same punctuation, spacing, line breaks
6. Return ONLY the processed text, NO explanations

Your job is to ACTIVELY find and replace 他/她 with names. Do NOT just return the original text."""


def split_into_chunks(text, target_size=500):
    """Split text by lines, accumulating until target_size is reached."""
    lines = text.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        line_with_newline = line + '\n'
        if len(current_chunk) > 0 and len(current_chunk) + len(line_with_newline) > target_size:
            chunks.append(current_chunk)
            current_chunk = line_with_newline
        else:
            current_chunk += line_with_newline

    if current_chunk:
        chunks.append(current_chunk.rstrip('\n'))

    if chunks and chunks[-1].endswith('\n'):
        chunks[-1] = chunks[-1].rstrip('\n')

    return chunks


def process_chunk(chunk, chunk_num, total_chunks):
    """Process a single chunk through the API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


def main():
    input_path = "data/cuotuo_suiyue.txt"
    output_path = "data/cuotuo_suiyue_resolved.txt"
    progress_path = "data/cuotuo_suiyue_resolved_progress.txt"

    print("Reading input file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Input text length: {len(text)} characters")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text, target_size=1000)
    print(f"Created {len(chunks)} chunks")

    with open(progress_path, 'w', encoding='utf-8') as f:
        f.write('')

    processed_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks"), 1):
        try:
            processed = process_chunk(chunk, i, len(chunks))
            processed_chunks.append(processed)
            with open(progress_path, 'a', encoding='utf-8') as f:
                f.write(processed)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            processed_chunks.append(chunk)
            with open(progress_path, 'a', encoding='utf-8') as f:
                f.write(chunk)

    print("\nReconstructing text...")
    resolved_text = ''.join(processed_chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(resolved_text)

    print(f"\nDone!")
    print(f"Final output saved to: {output_path}")
    print(f"Progress file: {progress_path}")
    print(f"Output text length: {len(resolved_text)} characters")


if __name__ == "__main__":
    main()
