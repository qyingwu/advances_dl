from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"  # Make sure this matches the name used in the grader
    model_path = Path(__file__).parent / model_name
    print(f"Model path: {model_path}")

    llm = BaseLLM()
    print(f"Base model loaded, device: {llm.device}")
    
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    print("LoRA adapters loaded")
    
    llm.model.eval()
    print("Model set to eval mode")
    
    # Set the model_name attribute to identify this as an SFT model
    llm.model_name = "sft"
    print("Model name set to 'sft'")

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair with explicit unit conversion hints.
    """
    # Round the answer to 3 decimal places for simplicity
    rounded_answer = round(float(answer), 3)
    
    # Format the question with explicit conversion instructions
    formatted_question = (
        f"{prompt}\n"
        "Follow these steps:\n"
        "1. Identify the units being converted\n"
        "2. Use the correct conversion factor\n"
        "3. Perform the calculation\n"
        "4. Round to 3 decimal places\n"
        "Please provide your answer in the format <answer>X</answer> where X is the numerical answer."
    )
    
    formatted_answer = f"<answer>{rounded_answer}</answer>"
    
    return {
        "question": formatted_question,
        "answer": formatted_answer
    }


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
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


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
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the base model
    llm = BaseLLM()
    
    # Configure LoRA with optimized parameters
    lora_config = LoraConfig(
        r=8,  # Increased rank for better capacity
        lora_alpha=32,  # Alpha = 4x rank
        target_modules="all-linear",
        lora_dropout=0.1,  # Add dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert the model to a LoRA model
    model = get_peft_model(llm.model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Prepare the dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Set up training arguments with optimized settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=2e-4,  # Higher learning rate with warm-up
        num_train_epochs=15,  # More epochs for better learning
        per_device_train_batch_size=8,  # Smaller batch size
        gradient_accumulation_steps=4,  # Effective batch size = 32
        max_grad_norm=0.3,  # Gradient clipping for stability
        gradient_checkpointing=True,
        save_strategy="epoch",
        weight_decay=0.05,  # Increased weight decay
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts",  # Cosine with restarts for better optimization
        save_total_limit=2,  # Keep only the last 2 checkpoints
        logging_steps=10,
        optim="adamw_torch",  # Use PyTorch's AdamW
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    
    # Test the model
    test_model(output_dir)
    
    print("\nTraining completed. To run the grader, use:")
    print("python -m grader.tests")


def test_model(ckpt_path: str):
    print("\nStarting test_model...")
    testset = Dataset("valid")
    print(f"Loaded validation dataset with {len(testset)} examples")
    
    print("\nLoading model...")
    llm = load_sft()
    print("Model loaded successfully")

    # Print a few example questions and answers
    print("\nTesting a few example questions:")
    for i in range(3):
        question = testset[i][0]
        correct_answer = testset[i][1]
        formatted_prompt = llm.format_prompt(question)
        print(f"\nQuestion {i+1}: {question}")
        print(f"Formatted prompt: {formatted_prompt}")
        
        raw_output = llm.generate(formatted_prompt)
        print(f"Raw output: {raw_output}")
        
        parsed_answer = llm.parse_answer(raw_output)
        print(f"Parsed answer: {parsed_answer}")
        print(f"Correct answer: {correct_answer}")
        print(f"Is correct: {abs(parsed_answer - correct_answer) < 0.05 * abs(correct_answer)}")

    # Run the benchmark
    print("\nRunning benchmark on validation set...")
    benchmark_result = benchmark(llm, testset, 100)
    print(f"\nBenchmark Results:")
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


def load_sft(ckpt_path: str = None) -> BaseLLM:
    print(f"Loading SFT model...")
    from pathlib import Path

    from peft import PeftModel

    if ckpt_path is None:
        ckpt_path = str(Path(__file__).parent / "sft_model")
    print(f"Using model path: {ckpt_path}")

    llm = BaseLLM()
    print(f"Base model loaded, device: {llm.device}")
    
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    print("LoRA adapters loaded")
    
    llm.model.eval()
    print("Model set to eval mode")
    
    # Set the model_name attribute to identify this as an SFT model
    llm.model_name = "sft"
    print("Model name set to 'sft'")

    return llm


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load_sft})