from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that performs unit conversions. Be concise and provide the answer in the format <answer>X</answer> where X is a number."
            },
            {
                "role": "user",
                "content": "Convert 2 hours to minutes."
            },
            {
                "role": "assistant",
                "content": "1 hour = 60 minutes. 2 * 60 = <answer>120</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
