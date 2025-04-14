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
        Format the prompt for the model.
        For CoT model, use the chat template.
        For SFT model, just return the question.
        """
        # Check if we're using the CoT model or the SFT model
        if hasattr(self, 'model_name') and self.model_name == 'cot':
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves unit conversion problems. Provide your reasoning and then give your answer in the format <answer>X</answer> where X is the numerical answer."},
                {"role": "user", "content": question}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # For SFT model, just return the question
            return question

    def parse_answer(self, text: str) -> float:
        """
        Parse the answer from the model's output.
        The answer should be in the format <answer>X</answer> where X is a number.
        If the answer is not in this format, try to find any number in the text.
        """
        # Try to find the answer in the format <answer>X</answer>
        answer_match = re.search(r'<answer>([\d.]+)</answer>', text)
        if answer_match:
            return float(answer_match.group(1))
        
        # If no answer tags found, try to find any number in the text
        numbers = re.findall(r'[\d.]+', text)
        if numbers:
            # Clean the number by removing any trailing non-numeric characters
            clean_number = re.sub(r'[^\d.]', '', numbers[0])
            try:
                return float(clean_number)
            except ValueError:
                return float('nan')
        
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
                    max_new_tokens=100,
                    do_sample=False,  # Use greedy decoding for stability
                    num_beams=1,  # Use greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Get the length of the input sequence for masking
            input_length = inputs["input_ids"].shape[1]
            
            # Decode only the generated tokens (excluding input tokens)
            generated_tokens = outputs[:, input_length:]
            decoded_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
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
            max_new_tokens=100,
            do_sample=False,  # Use greedy decoding for stability
            num_return_sequences=num_return_sequences if num_return_sequences is not None else 1,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Remove parameters that might cause instability
            # temperature=0.7,
            # top_p=0.9,
            # top_k=50,
            # repetition_penalty=1.2,
            # no_repeat_ngram_size=3,
            # early_stopping=True,
            # min_length=1,
            # max_length=200,
            # length_penalty=1.0,
            # bad_words_ids=[[self.tokenizer.pad_token_id]] if self.tokenizer.pad_token_id is not None else None,
            # use_cache=True
        )
        
        # Get the length of the first input sequence for masking
        input_length = inputs["input_ids"].shape[1]
        
        # Decode only the generated tokens (excluding input tokens)
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
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
