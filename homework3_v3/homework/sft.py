from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"  # Make sure this matches the name used in the grader
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    
    # Set the model_name attribute to identify this as an SFT model
    llm.model_name = "sft"

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
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round the answer to 3 decimal places for simplicity
    rounded_answer = round(float(answer), 3)
    
    # Format the question and answer
    formatted_question = prompt
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
    
    # Configure LoRA with higher rank for better accuracy
    lora_config = LoraConfig(
        r=24,  # Higher rank for better accuracy
        lora_alpha=96,  # Alpha parameter (4x rank)
        target_modules="all-linear",  # Apply LoRA to all linear layers
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert the model to a LoRA model
    model = get_peft_model(llm.model, lora_config)
    
    # Enable input require grads for gradient checkpointing
    model.enable_input_require_grads()
    
    # Prepare the dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Set up training arguments with improved settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=5e-5,  # Lower learning rate for stability
        num_train_epochs=10,  # More epochs for better accuracy
        per_device_train_batch_size=16,  # Smaller batch size for stability
        gradient_checkpointing=True,
        save_strategy="epoch",
        weight_decay=0.01,  # Add weight decay for regularization
        warmup_ratio=0.1,  # Add warmup
        lr_scheduler_type="cosine",  # Use cosine learning rate schedule
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
    
    # Print a message to remind the user to run the grader
    print("\nTraining completed. To run the grader, use:")
    print("python -m grader.tests")


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
