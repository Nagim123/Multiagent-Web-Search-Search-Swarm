import os
import backoff
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Initialize tokenizer and model for LLaMA-3-8B
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3-8b")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-8b")

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 15000

def completions_with_backoff(**kwargs):
    return generate_completion(**kwargs)

def generate_completion(model_name, prompt, temperature=1.0, max_tokens=100, n=1, stop=None):
    outputs = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_tokens,
                temperature=temperature,
                top_p=1.0,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Apply stop token logic
        if stop:
            for stop_token in stop:
                if stop_token in text:
                    text = text.split(stop_token)[0]
                    break
        
        outputs.append(text.strip())
    return outputs

def gpt3(prompt, model="text-davinci-002", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    return generate_completion(model, prompt, temperature, max_tokens, n, stop)

def gpt(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        return chatgpt([{"role": "user", "content": prompt}], model, temperature, max_tokens, n, stop)

def gpt4(prompt, model="gpt-4", temperature=0.2, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        return chatgpt([{"role": "user", "content": prompt}], model, temperature, max_tokens, n, stop)

def chatgpt(messages, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    prompt_text = " ".join([msg["content"] for msg in messages if msg["role"] == "user"])
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = generate_completion(model, prompt_text, temperature, max_tokens, cnt, stop)
        outputs.extend(res)
        # For simplicity, assuming token count approximation here as usage data is model-specific
        completion_tokens += max_tokens * len(res)
        prompt_tokens += len(prompt_text.split())
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}