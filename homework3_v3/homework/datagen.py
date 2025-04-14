from .cot import CoTModel
from .data import Dataset, is_answer_valid
import json
from pathlib import Path
import re
import random


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


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.8):
    """
    Generate a dataset for RFT by using the CoT model to generate multiple completions
    and selecting the ones with correct answers.
    
    Args:
        output_json: Path to the output JSON file
        oversample: Number of completions to generate for each question
        temperature: Temperature for generation
    """
    # Create the output directory if it doesn't exist
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the CoT model
    model = CoTModel()
    
    # Load the training dataset
    train_dataset = Dataset("train")
    total_questions = len(train_dataset)
    print(f"Loaded {total_questions} questions from training dataset")
    
    # Initialize the output dataset
    rft_dataset = []
    
    # Load existing dataset if it exists and is not empty
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, "r") as f:
                rft_dataset = json.load(f)
            print(f"Loaded {len(rft_dataset)} existing examples from {output_json}")
        except json.JSONDecodeError:
            rft_dataset = []
            print(f"Could not load existing dataset from {output_json}, starting fresh")
    
    # Process each question in the training dataset
    processed = 0
    skipped = 0
    added = 0
    
    for i, (question, correct_answer) in enumerate(train_dataset):
        processed += 1
        
        # Skip questions that are already in the dataset
        if any(item[0] == question for item in rft_dataset):
            skipped += 1
            if processed % 10 == 0:
                print(f"Progress: {processed}/{total_questions} questions processed, {skipped} skipped, {added} added")
            continue
        
        # Use the CoTModel's format_prompt method to create a properly formatted prompt
        formatted_prompt = model.format_prompt(question)
        
        # Generate multiple completions
        try:
            completions = model.batched_generate(
                [formatted_prompt] * oversample,
                num_return_sequences=oversample,
                temperature=temperature
            )
        except Exception as e:
            print(f"Error generating completions for question {i}: {e}")
            continue
        
        # Check each completion for correctness
        found_correct = False
        for i, completion in enumerate(completions):
            # Handle the case where completion is a list (when num_return_sequences > 1)
            if isinstance(completion, list):
                for j, sub_completion in enumerate(completion):
                    answer = extract_answer(sub_completion)
                    if answer is not None:
                        if is_answer_valid(answer, correct_answer):
                            rft_dataset.append([question, correct_answer, sub_completion])
                            found_correct = True
                            added += 1
                            break
                if found_correct:
                    break
            else:
                # Handle the case where completion is a string
                answer = extract_answer(completion)
                if answer is not None:
                    if is_answer_valid(answer, correct_answer):
                        rft_dataset.append([question, correct_answer, completion])
                        found_correct = True
                        added += 1
                        break
        
        # Print progress every 10 questions
        if processed % 10 == 0:
            print(f"Progress: {processed}/{total_questions} questions processed, {skipped} skipped, {added} added")
        
        # Write the dataset to the output JSON file after each question
        with open(output_path, "w") as f:
            json.dump(rft_dataset, f, indent=2)
    
    print(f"Dataset generation complete!")
    print(f"Total questions processed: {processed}")
    print(f"Questions skipped (already in dataset): {skipped}")
    print(f"New examples added: {added}")
    print(f"Total examples in dataset: {len(rft_dataset)}")
    
    return rft_dataset


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
