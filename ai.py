#Create an ai model that can be used to generate text based on a given prompt. The model should be able to understand the context of the prompt and generate coherent and relevant text.import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
class TextGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
# Example usage
if __name__ == "__main__":
    text_generator = TextGenerator()
    prompt = "Once upon a time in a land far away,"
    generated_text = text_generator.generate_text(prompt)
    print(generated_text)