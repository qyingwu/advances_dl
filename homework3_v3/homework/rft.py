from .base_llm import BaseLLM
from .data import Dataset, benchmark
import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def load_rft(ckpt_path: str = None, base_model: BaseLLM = None) -> BaseLLM:
    """
    Load a model from a checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint
        base_model: Base model to load the checkpoint into (optional)
        
    Returns:
        The loaded model
    """
    # If no path is provided, use the default path in the homework directory
    if ckpt_path is None:
        ckpt_path = str(Path(__file__).parent / "rft_model")
    
    print(f"RFT: Loading model from {ckpt_path}")
    from peft import PeftModel
    
    # If no base model is provided, create one
    if base_model is None:
        base_model = BaseLLM()
    
    # Load the model with LoRA adapters
    base_model.model = PeftModel.from_pretrained(base_model.model, ckpt_path).to(base_model.device)
    base_model.model.eval()  # Set to evaluation mode
    
    # Set the model_name attribute to identify this as an RFT model
    base_model.model_name = "rft"
    
    return base_model


# Alias for backward compatibility
def load(ckpt_path: str = None, base_model: BaseLLM = None) -> BaseLLM:
    """
    Alias for load_rft function for backward compatibility.
    """
    return load_rft(ckpt_path, base_model)

def format_example(prompt: str, answer: str, reasoning: str = None) -> dict[str, str]:
    """
    Construct a question / answer pair for RFT dataset.
    The RFT dataset has entries in the format [question, answer, reasoning].
    """
    try:
        # For RFT dataset, the prompt is already the question
        # If reasoning is provided, use it as the answer (it contains the <answer> tags)
        # Otherwise, format the answer with <answer> tags
        if reasoning is not None:
            # Validate that reasoning contains <answer> tags
            if "<answer>" not in reasoning or "</answer>" not in reasoning:
                print(f"Warning: Reasoning does not contain <answer> tags: {reasoning}")
                # Add <answer> tags if missing
                reasoning = f"<answer>{answer}</answer>"
            
            return {
                "question": prompt,
                "answer": reasoning
            }
        else:
            try:
                # Round the answer to 3 decimal places for simplicity
                rounded_answer = round(float(answer), 3)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert answer to float: {answer}")
                rounded_answer = 0.0
            
            # Format the question with explicit conversion instructions
            formatted_question = (
                f"{prompt}\n"
                "Follow these steps carefully:\n"
                "1. Identify the units being converted (source and target)\n"
                "2. For time units (years, months, weeks):\n"
                "   - Use 52.1775 weeks per year\n"
                "   - Use 4.34812 weeks per month\n"
                "   - For years to weeks: multiply by 52.1775\n"
                "   - For months to weeks: multiply by 4.34812\n"
                "3. For bit conversions (KB, MB, GB):\n"
                "   - Remember: 1 byte = 8 bits\n"
                "   - For KB to bits: multiply by 1024 * 8 = 8192\n"
                "   - For MB to bits: multiply by 1024 * 1024 * 8 = 8388608\n"
                "   - For GB to bits: multiply by 1024 * 1024 * 1024 * 8 = 8589934592\n"
                "   - For KB to bytes: multiply by 1024\n"
                "   - For MB to bytes: multiply by 1024 * 1024\n"
                "4. For complex conversions:\n"
                "   - Break down into multiple steps\n"
                "   - Convert through intermediate units if needed\n"
                "   - Use precise conversion factors\n"
                "   - Show all intermediate calculations\n"
                "5. Round the final answer to 3 decimal places\n"
                "Please provide your answer in the format <answer>X</answer> where X is the numerical answer."
            )
            
            formatted_answer = f"<answer>{rounded_answer}</answer>"
            
            return {
                "question": formatted_question,
                "answer": formatted_answer
            }
    except Exception as e:
        print(f"Error in format_example: {e}")
        # Return a valid example to prevent training from crashing
        return {
            "question": "Convert 1 meter to centimeters",
            "answer": "<answer>100</answer>"
        }

def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    try:
        # Ensure inputs are strings
        question = str(question)
        answer = str(answer)
        
        # Check for empty inputs
        if not question.strip() or not answer.strip():
            print(f"Warning: Empty input detected: question='{question}', answer='{answer}'")
            question = "Convert 1 meter to centimeters"
            answer = "<answer>100</answer>"
        
        full_text = f"{question} {answer}{tokenizer.eos_token}"

        # Set padding side to right for consistent tokenization
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize with error handling
        try:
            full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)
        except Exception as e:
            print(f"Error during tokenization: {e}")
            # Fallback to a simple tokenization
            full = tokenizer("Convert 1 meter to centimeters <answer>100</answer>", 
                            padding="max_length", truncation=True, max_length=128)

        input_ids = full["input_ids"]
        
        # Tokenize question separately
        try:
            question_tokens = tokenizer(question, padding="max_length", truncation=True, max_length=128)
            question_len = len(question_tokens["input_ids"])
        except Exception as e:
            print(f"Error tokenizing question: {e}")
            question_len = 10  # Fallback value
        
        # Create labels: mask out the prompt part
        labels = [-100] * question_len + input_ids[question_len:]

        # Ensure labels are properly masked
        for i in range(len(labels)):
            if i >= len(full["attention_mask"]) or full["attention_mask"][i] == 0:
                labels[i] = -100

        full["labels"] = labels
        return full
    except Exception as e:
        print(f"Error in tokenize function: {e}")
        # Return a valid tokenized example
        return tokenizer("Convert 1 meter to centimeters <answer>100</answer>", 
                        padding="max_length", truncation=True, max_length=128,
                        return_tensors="pt")


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Check if the data element has 3 arguments (question, answer, reasoning)
        if len(self.data[idx]) == 3:
            question, answer, reasoning = self.data[idx]
            formated_data = self.format_fn(question, answer, reasoning)
        else:
            # Handle the case where there are only 2 arguments
            formated_data = self.format_fn(*self.data[idx])
            
        return tokenize(self.tokenizer, **formated_data)
    

class RFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        print(f"\nInitializing RFTDataset with max_length={max_length}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the RFT dataset
        print(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} data items")
        
        # Process the data
        self.examples = []
        print("Processing data items...")
        for i, item in enumerate(self.data):
            try:
                # Validate data format
                if not isinstance(item, list) or len(item) < 3:
                    print(f"Warning: Skipping invalid data item at index {i}: {item}")
                    continue
                
                # RFT dataset has entries in the format [question, answer, reasoning]
                question = item[0]  # The question text
                answer = item[1]    # The numerical answer (not used directly)
                reasoning = item[2]  # The reasoning text that includes the answer in <answer> tags
                
                # Validate data types
                if not isinstance(question, str) or not isinstance(answer, (int, float, str)) or not isinstance(reasoning, str):
                    print(f"Warning: Skipping item with invalid data types at index {i}")
                    continue
                
                # Format the example - we use the reasoning as the answer because it contains
                # the formatted answer with <answer> tags
                example = format_example(question, answer, reasoning)
                self.examples.append(example)
            except Exception as e:
                print(f"Error processing data item at index {i}: {e}")
                continue
        
        print(f"Successfully processed {len(self.examples)} examples out of {len(self.data)} data items")
        
        # Print a sample example
        if self.examples:
            print("\nSample example:")
            sample = self.examples[0]
            print(f"Question: {sample['question'][:100]}...")
            print(f"Answer: {sample['answer'][:100]}...")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            # Format the input as "question\ncompletion"
            input_text = f"{example['question']}\n{example['answer']}"
            
            # Tokenize the input
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for now)
            labels = inputs["input_ids"].clone()
            
            # Tokenize just the question to find where it ends
            question_tokens = self.tokenizer(
                example["question"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Find the actual length of the question (excluding padding)
            question_length = 0
            for i, token_id in enumerate(question_tokens["input_ids"][0]):
                if token_id == self.tokenizer.pad_token_id:
                    break
                question_length = i + 1
            
            # Set labels to -100 for the question part and padding
            labels[0, :question_length] = -100  # Question part
            labels[0, question_length:] = inputs["input_ids"][0, question_length:]  # Answer part
            
            # Also mask out padding tokens
            labels[0, inputs["attention_mask"][0] == 0] = -100
            
            # Ensure all tensors are on CPU and are the correct type
            result = {
                "input_ids": inputs["input_ids"].squeeze().to(torch.long),
                "attention_mask": inputs["attention_mask"].squeeze().to(torch.long),
                "labels": labels.squeeze().to(torch.long)
            }
            
            # Debug info for the first few items
            if idx < 3:
                print(f"\nItem {idx} debug info:")
                print(f"Input text length: {len(input_text)}")
                print(f"Question length: {question_length}")
                print(f"Input IDs shape: {result['input_ids'].shape}")
                print(f"Labels shape: {result['labels'].shape}")
                print(f"Labels range: {result['labels'].min()} to {result['labels'].max()}")
                print(f"Number of -100 in labels: {(result['labels'] == -100).sum().item()}")
                print(f"Number of non -100 in labels: {(result['labels'] != -100).sum().item()}")
            
            return result
        except Exception as e:
            print(f"Error processing example at index {idx}: {e}")
            # Return a valid example to prevent training from crashing
            if len(self.examples) > 0:
                # Use the first example as a fallback
                return self.__getitem__(0)
            else:
                # If no examples are available, create a minimal valid example
                return {
                    "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                    "labels": torch.full((self.max_length,), -100, dtype=torch.long)
                }

def train_model(
    output_dir: str,
    **kwargs,
):
    """
    Train a model using LoRA fine-tuning with improved settings.
    
    Args:
        output_dir: Directory to save the model checkpoints
        **kwargs: Additional arguments for training
    """
    from pathlib import Path
    import torch
    from transformers import TrainingArguments, Trainer, get_scheduler
    from peft import get_peft_model, LoraConfig, TaskType
    
    print("\n" + "="*50)
    print("STARTING RFT TRAINING")
    print("="*50)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the base model
    print("\nLoading base model...")
    llm = BaseLLM()
    print(f"Base model loaded")
    print(f"Model device: {llm.device}")
    
    # Configure LoRA with optimized parameters
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=16,  # Keep rank at 16
        lora_alpha=32,  # Alpha = 2x rank
        target_modules="all-linear",
        lora_dropout=0.1,  # Increased dropout for better regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print(f"LoRA config: {lora_config}")
    
    # Convert the model to a LoRA model
    print("\nConverting model to LoRA...")
    model = get_peft_model(llm.model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    print("\nEnabling gradient checkpointing...")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Prepare the dataset
    print("\nPreparing dataset...")
    data_path = os.path.join(Path(__file__).parent.parent, "data", "rft.json")
    print(f"Loading data from: {data_path}")
    train_dataset = RFTDataset(data_path, llm.tokenizer)
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Print a sample from the dataset
    print("\nSample from dataset:")
    sample = train_dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Labels range: {sample['labels'].min()} to {sample['labels'].max()}")
    print(f"Number of -100 in labels: {(sample['labels'] == -100).sum().item()}")
    
    # Set up training arguments with optimized settings
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=5e-5,  # Lower learning rate for better stability
        num_train_epochs=25,  # Increased epochs for better learning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Increased for larger effective batch size
        max_grad_norm=0.5,  # Reduced for more stable gradients
        gradient_checkpointing=True,
        save_strategy="epoch",
        weight_decay=0.05,  # Increased weight decay for better regularization
        warmup_ratio=0.15,  # Longer warmup
        lr_scheduler_type="cosine_with_restarts",  # Added restarts for better optimization
        save_total_limit=2,
        logging_steps=5,
        optim="adamw_torch",
        label_names=["labels"],
        fp16=False,
    )
    print(f"Training arguments: {training_args}")
    
    # Create the trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    
    # Test the model
    print("\nTesting model...")
    test_model(output_dir)
    
    print("\nTraining completed. To run the grader, use:")
    print("python -m grader.tests")
    print("="*50)


def test_model(model_path: str, question: str | None = None):
    """
    Test the RFT model on a single question or the full benchmark.
    """
    try:
        # Load the model with LoRA adapters
        llm = load_rft(model_path)
        
        if question is not None:
            # Test on a single question
            print(f"\nTesting question: {question}")
            formatted_prompt = llm.format_prompt(question)
            print(f"Formatted prompt: {formatted_prompt}")
            
            raw_output = llm.generate(formatted_prompt)
            print(f"Raw output: {raw_output}")
            
            answer = llm.parse_answer(raw_output)
            print(f"Parsed answer: {answer}")
            
            return answer
        
        # Run full benchmark with a mock dataset
        print("\nRunning benchmark...")
        mock_data = [
            ("Convert 5 years to weeks", 260.0),
            ("Convert 10 kilograms to pounds", 22.0462),
            ("Convert 2 GB to MB", 2048.0),
            ("Convert 2 KB to bits", 16384.0),
            ("Convert 5 quarts to pints", 10.0)
        ]
        testset = type('MockDataset', (), {
            '__len__': lambda self: len(mock_data),
            '__getitem__': lambda self, idx: mock_data[idx]
        })()
        
        benchmark_result = benchmark(llm, testset, max_question=5)
        
        # Print detailed results
        print("\nBenchmark Results:")
        print(f"Accuracy: {benchmark_result.accuracy:.4f}")
        print(f"Answer Rate: {benchmark_result.answer_rate:.4f}")
        
        # Print a few example predictions
        print("\nExample Predictions:")
        for i, sample in enumerate(benchmark_result.samples[:5]):
            print(f"\nExample {i+1}:")
            print(f"Question: {sample.question}")
            print(f"Predicted: {sample.answer}")
            print(f"Correct: {sample.correct_answer}")
            print(f"Is Correct: {sample.is_correct}")
        
        return benchmark_result
    
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Try using the model directly with generate.py for more detailed error information.")
        return None

def test_samples():
    """
    Test specific samples with the RFT model to analyze performance, including edge cases.
    """
    print("\nTesting RFT model on specific samples and edge cases...")
    
    # Load the model
    llm = load_rft()
    print("Model loaded successfully")
    
    # Test cases with expected answers, including edge cases
    test_cases = [
        # Basic conversions (for baseline)
        ("Convert 2 years to weeks", 104.355),
        ("Convert 5 GB to MB", 5000.0),
        
        # Edge cases - very small numbers
        ("Convert 0.001 seconds to milliseconds", 1.0),
        ("Convert 0.0001 meters to millimeters", 0.1),
        
        # Edge cases - very large numbers
        ("Convert 1000 terabytes to gigabytes", 1000000.0),
        ("Convert 1000 light years to kilometers", 9.461e15),
        
        # Edge cases - zero and negative values
        ("Convert 0 hours to minutes", 0.0),
        ("Convert -5 degrees Celsius to Fahrenheit", 23.0),
        
        # Complex multi-step conversions
        ("Convert 1 millennium to hours", 8766000.0),
        ("Convert 1 light year to miles", 5.878e12),
        
        # Uncommon unit conversions
        ("Convert 1 parsec to light years", 3.26156),
        ("Convert 1 astronomical unit to kilometers", 149597870.7),
        
        # Precision-sensitive conversions
        ("Convert 1 pound to grams", 453.592),
        ("Convert 1 mile to meters", 1609.344),
        
        # Common unit conversions (to test consistency)
        ("Convert 1 hour to minutes", 60.0),
        ("Convert 1 kilometer to meters", 1000.0),
        ("Convert 1 kilogram to grams", 1000.0),
        ("Convert 1 liter to milliliters", 1000.0),
        ("Convert 2 months to days", 60.0),
        
        # Decimal handling
        ("Convert 2.5 hours to minutes", 150.0),
        ("Convert 1.5 kilometers to meters", 1500.0),
        ("Convert 0.75 kilograms to grams", 750.0),
        
        # Format variations
        ("How many weeks are in 2 years?", 104.355),
        ("What is 5 GB in MB?", 5000.0),
        ("Convert five gigabytes to megabytes", 5000.0),
        
        # Edge cases - unit spelling variations
        ("Convert 1 metre to centimetres", 100.0),  # British spelling
        ("Convert 1 meter to centimeters", 100.0),  # American spelling
        
        # Edge cases - spacing and formatting
        ("Convert1hourto minutes", 60.0),  # No spaces
        ("   Convert   1   hour   to   minutes   ", 60.0),  # Extra spaces
        
        # Edge cases - different number formats
        ("Convert 1,000 meters to kilometers", 1.0),  # Comma in number
        ("Convert 1e3 meters to kilometers", 1.0),    # Scientific notation
        
        # Edge cases - compound units
        ("Convert 60 miles per hour to kilometers per hour", 96.56),
        ("Convert 1 meter per second to kilometers per hour", 3.6),
        
        # Edge cases - temperature conversions (non-linear)
        ("Convert 0 Celsius to Fahrenheit", 32.0),
        ("Convert 100 Celsius to Fahrenheit", 212.0),
        ("Convert -40 Celsius to Fahrenheit", -40.0),  # Interesting case where they're equal
        
        # Edge cases - precision requirements
        ("Convert 1/3 hour to minutes", 20.0),
        ("Convert pi kilometers to meters", 3141.59),  # Testing mathematical constants
    ]
    
    print("\nTesting individual questions:")
    total_tests = len(test_cases)
    passed_tests = 0
    failed_cases = []
    
    for question, expected in test_cases:
        print(f"\nQuestion: {question}")
        print(f"Expected answer: {expected}")
        
        # Format and generate
        formatted_prompt = llm.format_prompt(question)
        print(f"Formatted prompt: {formatted_prompt}")
        
        raw_output = llm.generate(formatted_prompt)
        print(f"Raw output: {raw_output}")
        
        # Parse and validate
        try:
            answer = llm.parse_answer(raw_output)
            print(f"Parsed answer: {answer}")
            
            # Check if answer is within 5% of expected
            is_correct = abs(answer - expected) / expected < 0.05 if expected != 0 else abs(answer) < 1e-6
            print(f"Is correct (within 5%): {is_correct}")
            
            if is_correct:
                passed_tests += 1
            else:
                relative_error = abs(answer - expected) / expected * 100 if expected != 0 else abs(answer)
                print(f"Relative error: {relative_error:.2f}%")
                failed_cases.append((question, expected, answer, relative_error))
        except Exception as e:
            print(f"Error parsing answer: {e}")
            failed_cases.append((question, expected, "ERROR", None))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.2f}%")
    
    if failed_cases:
        print("\nFailed cases:")
        for question, expected, actual, error in failed_cases:
            print(f"\nQuestion: {question}")
            print(f"Expected: {expected}")
            print(f"Got: {actual}")
            if error is not None:
                print(f"Relative error: {error:.2f}%")
    
    # Test batch processing with a subset of cases
    print("\nTesting batch processing with a subset of cases:")
    subset_cases = test_cases[:5]  # Take first 5 cases for batch testing
    questions = [case[0] for case in subset_cases]
    expected = [case[1] for case in subset_cases]
    
    try:
        answers = llm.answer(*questions)
        print("\nBatch results:")
        batch_passed = 0
        for q, a, e in zip(questions, answers, expected):
            print(f"\nQuestion: {q}")
            print(f"Expected: {e}")
            print(f"Got: {a}")
            is_correct = abs(a - e) / e < 0.05 if e != 0 else abs(a) < 1e-6
            print(f"Is correct (within 5%): {is_correct}")
            if is_correct:
                batch_passed += 1
            else:
                relative_error = abs(a - e) / e * 100 if e != 0 else abs(a)
                print(f"Relative error: {relative_error:.2f}%")
        
        print(f"\nBatch processing success rate: {(batch_passed/len(subset_cases))*100:.2f}%")
    except Exception as e:
        print(f"Error in batch processing: {e}")

if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load_rft, "samples": test_samples})