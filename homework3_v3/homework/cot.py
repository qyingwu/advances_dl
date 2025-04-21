from .base_llm import BaseLLM
from .data import Dataset, benchmark

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        return (
            "You are a helpful assistant for unit conversion tasks. Always show reasoning and format final answer as <answer>X.000</answer>.\n\n"
            "Examples:\n"
            "User: Convert 2 hours to minutes.\n"
            "Assistant: 1 hour = 60 minutes. 2 * 60 = <answer>120.000</answer>\n"
            "User: Convert 2 GB to MB.\n"
            "Assistant: 1 GB = 1000 MB. 2 * 1000 = <answer>2000.000</answer>\n"
            "User: Convert 10 kg to pounds.\n"
            "Assistant: 1 kg = 2.20462 pounds. 10 * 2.20462 = <answer>22.046</answer>\n"
            "User: Convert 0.001 seconds to milliseconds.\n"
            "Assistant: 1 second = 1000 milliseconds. 0.001 * 1000 = <answer>1.0</answer>\n"
            "User: Convert -5 degrees Celsius to Fahrenheit.\n"
            "Assistant: F = (C * 9/5) + 32 = (-5 * 9/5) + 32 = -9 + 32 = <answer>23.0</answer>\n"
            "User: Convert 1 meter per second to kilometers per hour.\n"
            "Assistant: 1 m/s = 3.6 km/h = <answer>3.6</answer>\n"
            f"User: {question}\n"
            "Assistant:"
        )


def load() -> CoTModel:
    return CoTModel()

def test_model(max_questions=10):
    """
    Test the CoT model on the validation dataset.
    
    Args:
        max_questions: Maximum number of questions to test on
    """
    import gc
    import torch

    testset = Dataset("valid")
    model = CoTModel()
    model.model.eval()  # Set to eval mode for inference

    # Use the correct parameter for benchmark
    benchmark_result = benchmark(model, testset, max_questions)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
