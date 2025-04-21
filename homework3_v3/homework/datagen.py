from .cot import CoTModel
from .data import Dataset, is_answer_valid
import json
from pathlib import Path
import re
import os


def extract_answer(text):
    """Extract the answer from the text using regex."""
    match = re.search(r'<answer>([\d.]+)(?:\.\.\.)?</answer>', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            # If conversion fails, try to extract just the numeric part
            numeric_part = re.search(r'[\d.]+', match.group(1))
            if numeric_part:
                try:
                    return float(numeric_part.group(0))
                except ValueError:
                    return None
            return None
    return None


def generate_complex_examples():
    """Generate additional examples of complex conversions."""
    examples = []
    
    # Time unit conversions
    time_conversions = [
        (
            "Convert 2 years to weeks",
            104.355,
            "1 year = 52.1775 weeks. 2 years = 2 * 52.1775 = <answer>104.355</answer> weeks"
        ),
        (
            "Convert 3 months to weeks",
            13.04436,
            "1 month = 4.34812 weeks. 3 months = 3 * 4.34812 = <answer>13.04436</answer> weeks"
        ),
        (
            "Convert 5 decades to weeks",
            2608.875,
            "1 decade = 10 years. 1 year = 52.1775 weeks. 5 decades = 5 * 10 * 52.1775 = <answer>2608.875</answer> weeks"
        ),
        (
            "Convert 1 century to weeks",
            5217.75,
            "1 century = 100 years. 1 year = 52.1775 weeks. 1 century = 100 * 52.1775 = <answer>5217.75</answer> weeks"
        ),
        (
            "Convert 8 years to months",
            96.0,
            "1 year = 12 months. 8 years = 8 * 12 = <answer>96</answer> months"
        ),
    ]
    
    # Bit-level conversions with detailed reasoning
    bit_conversions = [
        (
            "Convert 2 KB to bits",
            16384.0,
            "1 KB = 1024 bytes. 1 byte = 8 bits. 2 KB = 2 * 1024 * 8 = 2 * 8192 = <answer>16384</answer> bits"
        ),
        (
            "Convert 3 MB to bits",
            25165824.0,
            "1 MB = 1024 KB. 1 KB = 1024 bytes. 1 byte = 8 bits. 3 MB = 3 * 1024 * 1024 * 8 = 3 * 8388608 = <answer>25165824</answer> bits"
        ),
        (
            "Convert 1 GB to bits",
            8589934592.0,
            "1 GB = 1024 MB. 1 MB = 1024 KB. 1 KB = 1024 bytes. 1 byte = 8 bits. 1 GB = 1024 * 1024 * 1024 * 8 = <answer>8589934592</answer> bits"
        ),
        (
            "Convert 4 KB to bytes",
            4096.0,
            "1 KB = 1024 bytes. 4 KB = 4 * 1024 = <answer>4096</answer> bytes"
        ),
        (
            "Convert 2 MB to bytes",
            2097152.0,
            "1 MB = 1024 KB. 1 KB = 1024 bytes. 2 MB = 2 * 1024 * 1024 = <answer>2097152</answer> bytes"
        ),
        (
            "Convert 1 KB to bits",
            8192.0,
            "1 KB = 1024 bytes. 1 byte = 8 bits. 1 KB = 1024 * 8 = <answer>8192</answer> bits"
        ),
        (
            "Convert 5 MB to bits",
            41943040.0,
            "1 MB = 1024 KB. 1 KB = 1024 bytes. 1 byte = 8 bits. 5 MB = 5 * 1024 * 1024 * 8 = 5 * 8388608 = <answer>41943040</answer> bits"
        ),
        (
            "Convert 3 GB to bytes",
            3221225472.0,
            "1 GB = 1024 MB. 1 MB = 1024 KB. 1 KB = 1024 bytes. 3 GB = 3 * 1024 * 1024 * 1024 = <answer>3221225472</answer> bytes"
        ),
    ]
    
    # Complex multi-step conversions
    complex_conversions = [
        (
            "Convert 5 years to hours",
            43829.1,
            "1 year = 365.25 days. 1 day = 24 hours. 5 years = 5 * 365.25 * 24 = <answer>43829.1</answer> hours"
        ),
        (
            "Convert 2 decades to days",
            7304.843975625,
            "1 decade = 10 years. 1 year = 365.25 days. 2 decades = 2 * 10 * 365.25 = <answer>7304.843975625</answer> days"
        ),
        (
            "Convert 3 centuries to months",
            3600.0,
            "1 century = 100 years. 1 year = 12 months. 3 centuries = 3 * 100 * 12 = <answer>3600</answer> months"
        ),
        (
            "Convert 1 millennium to weeks",
            52177.45696874999,
            "1 millennium = 1000 years. 1 year = 52.17745696875 weeks. 1 millennium = 1000 * 52.17745696875 = <answer>52177.45696874999</answer> weeks"
        ),
        (
            "Convert 4 years to minutes",
            2103796.8,
            "1 year = 365.25 days. 1 day = 24 hours. 1 hour = 60 minutes. 4 years = 4 * 365.25 * 24 * 60 = <answer>2103796.8</answer> minutes"
        ),
    ]
    
    examples.extend(time_conversions)
    examples.extend(bit_conversions)
    examples.extend(complex_conversions)
    
    return examples


def batched_generate(model, prompts, temperature=0.7, max_new_tokens=100):
    """
    Generate completions for a batch of prompts.
    
    Args:
        model: The model to use for generation
        prompts: List of prompts to generate completions for
        temperature: Temperature for generation
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        List of completions
    """
    completions = []
    for prompt in prompts:
        try:
            formatted_prompt = model.format_prompt(prompt)
            completion = model.generate(formatted_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            completions.append(completion)
        except Exception as e:
            print(f"Error generating completion: {e}")
            completions.append("")
    return completions


def generate_dataset(
    output_json: str = "data/rft.json",
    oversample: bool = True,
    temperature: float = 0.7,
):
    """
    Generate a dataset for RFT by using the CoT model to generate multiple completions
    and selecting the ones with correct answers.
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Whether to oversample complex examples
        temperature: Base temperature for generation
    """
    # Create output directory if it doesn't exist
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize the CoT model
    cot_model = CoTModel()
    
    # Load existing examples if the file exists
    examples = []
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} existing examples from {output_json}")
        except json.JSONDecodeError:
            print(f"Error loading {output_json}, starting fresh")
    
    # Generate complex examples
    print("Generating complex examples...")
    complex_examples = generate_complex_examples()
    examples.extend(complex_examples)
    print(f"Added {len(complex_examples)} complex examples")
    
    # Process questions from the training dataset
    print("Processing questions from training dataset...")
    processed = 0
    skipped = 0
    added = 0
    
    # Use different temperatures for diversity
    temperatures = [0.3, 0.5, 0.7, 0.9]
    
    for question, answer in train_data:
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(train_data)} questions")
        
        # Skip if we already have this question
        if any(ex[0] == question for ex in examples):
            skipped += 1
            continue
        
        # Try different temperatures
        for temp in temperatures:
            # Generate multiple completions with different temperatures
            completions = batched_generate(
                cot_model, 
                [question] * 3,  # Generate 3 completions per question
                temperature=temp
            )
            
            # Find completions with correct answers
            for completion in completions:
                try:
                    # Extract the answer from the completion
                    parsed_answer = cot_model.parse_answer(completion)
                    
                    # Check if the answer is correct (within 1% error)
                    if abs(parsed_answer - answer) / answer < 0.01:
                        # Extract the reasoning (everything before the answer)
                        reasoning = completion.split("<answer>")[0].strip()
                        
                        # Add the example
                        examples.append([question, answer, reasoning])
                        added += 1
                        break  # Stop after finding one correct completion
                except:
                    continue
            
            # If we found a correct completion, stop trying temperatures
            if any(ex[0] == question for ex in examples):
                break
    
    print(f"Processed {processed} questions")
    print(f"Skipped {skipped} questions (already in dataset)")
    print(f"Added {added} new examples")
    print(f"Total dataset size: {len(examples)} examples")
    
    # Save the dataset
    with open(output_json, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved dataset to {output_json}")
    return examples


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
