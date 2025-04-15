from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Format a question for the model.
        For SFT model, we just return the question as is.
        For RFT model, we add instructions for reasoning.
        """
        if hasattr(self, 'model_name') and self.model_name == "rft":
            return f"{question}\nPlease provide your reasoning and then give your answer in the format <answer>X</answer> where X is the numerical answer."
        elif hasattr(self, 'model_name') and self.model_name == "sft":
            # For SFT model, add a hint to format the answer correctly
            return f"{question}\nPlease provide your answer in the format <answer>X</answer> where X is the numerical answer."
        else:
            # For other models, just return the question
            return question

    def parse_answer(self, text: str) -> float:
        """
        Parse the answer from the model's output.
        The answer should be in the format <answer>X</answer> where X is a number.
        If the answer is not in this format, try to find any number in the text.
        Returns NaN if no valid number is found.
        """
        try:
            # Try to find the answer in the format <answer>X</answer>
            answer_match = re.search(r'<answer>([\d.]+)</answer>', text)
            if answer_match:
                try:
                    # Clean the number and convert to float
                    clean_number = re.sub(r'[^\d.]', '', answer_match.group(1))
                    if clean_number:
                        # Handle numbers with multiple decimal points by keeping only the first one
                        if clean_number.count('.') > 1:
                            parts = clean_number.split('.')
                            clean_number = parts[0] + '.' + ''.join(parts[1:])
                        return round(float(clean_number), 3)
                except ValueError:
                    print(f"Warning: Could not convert '{answer_match.group(1)}' to float")
            
            # If no answer tags found or conversion failed, try to find any number in the text
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                # Try each number found until we get a valid float
                for num in numbers:
                    try:
                        # Clean the number by removing any trailing non-numeric characters
                        clean_number = re.sub(r'[^\d.]', '', num)
                        if clean_number:
                            # Handle numbers with multiple decimal points by keeping only the first one
                            if clean_number.count('.') > 1:
                                parts = clean_number.split('.')
                                clean_number = parts[0] + '.' + ''.join(parts[1:])
                            return round(float(clean_number), 3)
                    except ValueError:
                        continue
            
            print(f"Warning: No valid number found in text: '{text}'")
            return float('nan')
        except Exception as e:
            print(f"Error parsing answer: {str(e)}")
            return float('nan')

    def generate(self, prompt: str) -> str:
        """
        Generate text from the model with stable parameters.
        """
        try:
            # Set padding side to left for correct alignment
            self.tokenizer.padding_side = "left"
            
            # Tokenize the prompt with padding
            inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
            
            # Use stable generation parameters
            with torch.no_grad():  # Disable gradient computation
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,  # Reduced max tokens
                    do_sample=False,  # Use greedy decoding for stability
                    num_beams=1,  # Use greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Use KV cache for faster generation
                )
            
            # Get the length of the input sequence for masking
            input_length = inputs["input_ids"].shape[1]
            
            # Decode only the generated tokens (excluding input tokens)
            generated_tokens = outputs[:, input_length:]
            decoded_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # For SFT model, ensure the output is in the expected format
            if hasattr(self, 'model_name') and self.model_name == "sft":
                # Check if the output already contains <answer> tags
                if "<answer>" not in decoded_output:
                    # Try to find a number in the output
                    numbers = re.findall(r'[\d.]+', decoded_output)
                    if numbers:
                        # Clean the number by removing any trailing non-numeric characters
                        clean_number = re.sub(r'[^\d.]', '', numbers[0])
                        try:
                            # Format the output with <answer> tags
                            decoded_output = f"<answer>{clean_number}</answer>"
                        except ValueError:
                            pass
            
            return decoded_output
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"Error: {str(e)}"

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of generate method with stable parameters.
        """
        # Set padding side to left for correct alignment
        self.tokenizer.padding_side = "left"
        
        # Tokenize all prompts with padding
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        # Use more stable generation parameters
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,  # Reduced max tokens
            do_sample=num_return_sequences is not None and num_return_sequences > 1,  # Use sampling for multiple sequences
            temperature=temperature if num_return_sequences is not None and num_return_sequences > 1 else 1.0,  # Apply temperature only when sampling
            num_return_sequences=num_return_sequences if num_return_sequences is not None else 1,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True  # Use KV cache for faster generation
        )
        
        # Get the length of the first input sequence for masking
        input_length = inputs["input_ids"].shape[1]
        
        # Decode only the generated tokens (excluding input tokens)
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # For SFT model, ensure the outputs are in the expected format
        if hasattr(self, 'model_name') and self.model_name == "sft":
            for i in range(len(decoded_outputs)):
                # Check if the output already contains <answer> tags
                if "<answer>" not in decoded_outputs[i]:
                    # Try to find a number in the output
                    numbers = re.findall(r'[\d.]+', decoded_outputs[i])
                    if numbers:
                        # Clean the number by removing any trailing non-numeric characters
                        clean_number = re.sub(r'[^\d.]', '', numbers[0])
                        try:
                            # Format the output with <answer> tags
                            decoded_outputs[i] = f"<answer>{clean_number}</answer>"
                        except ValueError:
                            pass
        
        # Reshape output if num_return_sequences is specified
        if num_return_sequences is not None:
            return [decoded_outputs[i:i + num_return_sequences] for i in range(0, len(decoded_outputs), num_return_sequences)]
        
        return decoded_outputs

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
