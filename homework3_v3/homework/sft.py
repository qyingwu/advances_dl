from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load_sft() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
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
    # Round the answer to 3 decimal places
    rounded_answer = round(float(answer), 3)
    
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
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    llm = BaseLLM()
    
    
    lora_config = LoraConfig(
        r=8,  # lora rank = 8
        lora_alpha=32,  # Alpha = 4x rank
        target_modules="all-linear",
        lora_dropout=0.1, 
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert the model to a LoRA model
    model = get_peft_model(llm.model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=2e-4,
        num_train_epochs=15,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        save_strategy="epoch",
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=2,
        logging_steps=10,
        optim="adamw_torch",
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)

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

    print("\nRunning benchmark on validation set...")
    benchmark_result = benchmark(llm, testset, 100)
    print(f"\nBenchmark Results:")
    print(f"Accuracy: {benchmark_result.accuracy:.4f}")
    print(f"Answer Rate: {benchmark_result.answer_rate:.4f}")
    
    print("\nExample Predictions:")
    for i, sample in enumerate(benchmark_result.samples[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {sample.question}")
        print(f"Predicted: {sample.answer}")
        print(f"Correct: {sample.correct_answer}")
        print(f"Is Correct: {sample.is_correct}")


def load() -> BaseLLM:
    """
    Alias for load_sft function for backward compatibility.
    """
    return load_sft()


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load_sft})