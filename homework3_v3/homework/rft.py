from .base_llm import BaseLLM
from .data import Dataset, benchmark
import json
import os


def load(ckpt_path: str, base_model: BaseLLM = None) -> BaseLLM:
    """
    Load a model from a checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint
        base_model: Base model to load the checkpoint into (optional)
        
    Returns:
        The loaded model
    """
    from peft import PeftModel
    
    # If no base model is provided, create one
    if base_model is None:
        base_model = BaseLLM()
    
    # Load the model with LoRA adapters
    base_model.model = PeftModel.from_pretrained(base_model.model, ckpt_path).to(base_model.device)
    base_model.model.eval()  # Set to evaluation mode
    
    return base_model

def format_example(prompt: str, answer: str, reasoning: str = None) -> dict[str, str]:
    """
    Construct a question / answer pair for RFT dataset.
    The RFT dataset has entries in the format [question, answer, reasoning].
    """
    # For RFT dataset, the prompt is already the question
    # If reasoning is provided, use it as the answer (it contains the <answer> tags)
    # Otherwise, format the answer with <answer> tags
    if reasoning is not None:
        return {
            "question": prompt,
            "answer": reasoning
        }
    else:
        # Round the answer to 3 decimal places for simplicity
        rounded_answer = round(float(answer), 3)
        formatted_answer = f"<answer>{rounded_answer}</answer>"
        
        return {
            "question": prompt,
            "answer": formatted_answer
        }

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
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the RFT dataset
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        # Process the data
        self.examples = []
        for item in self.data:
            # RFT dataset has entries in the format [question, answer, reasoning]
            question = item[0]  # The question text
            answer = item[1]    # The numerical answer (not used directly)
            reasoning = item[2]  # The reasoning text that includes the answer in <answer> tags
            
            # Format the example - we use the reasoning as the answer because it contains
            # the formatted answer with <answer> tags
            example = format_example(question, answer, reasoning)
            self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
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
        
        # Mask out the question part in the labels
        question_tokens = self.tokenizer(
            example["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Set labels to -100 for the question part (these will be ignored in the loss)
        labels[:, :question_tokens["input_ids"].shape[1]] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
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
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the base model
    llm = BaseLLM()
    
    # Configure LoRA with higher rank for RFT (more complex task)
    lora_config = LoraConfig(
        r=32,  # Higher rank for better capacity for complex reasoning
        lora_alpha=128,  # Alpha parameter (4x rank)
        target_modules="all-linear",  # Apply LoRA to all linear layers
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert the model to a LoRA model
    model = get_peft_model(llm.model, lora_config)
    
    # Enable input require grads for gradient checkpointing
    model.enable_input_require_grads()
    
    # Prepare the dataset
    data_path = os.path.join(Path(__file__).parent.parent, "data", "rft.json")
    train_dataset = RFTDataset(data_path, llm.tokenizer)
    
    # Set up training arguments with improved settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=5e-5,  # Lower learning rate for stability
        num_train_epochs=3,  # Fewer epochs to prevent overfitting
        per_device_train_batch_size=8,  # Smaller batch size for stability
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
        train_dataset=train_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    
    # Test the model
    test_model(output_dir)


def test_model(model_path: str, question: str | None = None):
    """
    Test the RFT model on a single question or the full benchmark.
    """
    try:
        # Load the model with LoRA adapters
        llm = load(model_path)
        
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
        
        # Run full benchmark
        print("\nRunning benchmark...")
        results = benchmark(llm)
        
        # Print detailed results
        print("\nBenchmark Results:")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Answer Rate: {results['answer_rate']:.2%}")
        print(f"Total Samples: {results['total']}")
        print(f"Correct Answers: {results['correct']}")
        print(f"Valid Answers: {results['valid']}")
        
        # Print a few example predictions
        print("\nExample Predictions:")
        for i, (q, pred, true, is_correct) in enumerate(zip(
            results['questions'][:5],
            results['predictions'][:5],
            results['true_values'][:5],
            results['correctness'][:5]
        )):
            print(f"\nExample {i+1}:")
            print(f"Question: {q}")
            print(f"Predicted: {pred}")
            print(f"True Value: {true}")
            print(f"Correct: {is_correct}")
        
        return results
    
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Try using the model directly with generate.py for more detailed error information.")
        return None


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
