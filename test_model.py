from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen2.5-3B-Instruct")

prompt = "What is the vanishing gradient problem?"

messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=True)