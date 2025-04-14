import argparse
from homework.sft import load
from homework.base_llm import BaseLLM

def main():
    parser = argparse.ArgumentParser(description='Test the SFT model on example questions')
    args = parser.parse_args()
    
    # Load the trained SFT model
    model = load()
    
    # Example questions
    questions = [
        "Convert 5 meters to feet.",
        "Convert 10 kilometers to miles.",
        "Convert 2.5 pounds to kilograms.",
        "Convert 100 degrees Celsius to Fahrenheit.",
        "Convert 3 gallons to liters."
    ]
    
    print("Testing SFT model on example questions:")
    print("-" * 50)
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # Format the prompt
        formatted_prompt = model.format_prompt(question)
        print("\nFormatted prompt:")
        print(formatted_prompt)
        
        # Generate the answer
        raw_output = model.generate(formatted_prompt)
        print("\nRaw output:")
        print(raw_output)
        
        # Parse the answer
        answer = model.parse_answer(raw_output)
        print(f"\nParsed answer: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main() 