import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CyphaNLP:
    def __init__(self, model_name='gpt2', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.context = ""
    def chat(self, prompt: str, max_new_tokens=64, system_message: str = ""):
        ctx = (system_message+"\n" if system_message else "") + self.context + prompt
        input_ids = self.tokenizer.encode(ctx, return_tensors='pt').to(self.device)
        output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=40)
        resp = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        self.context += prompt + resp
        return resp
    def summarize(self, text: str, max_tokens: int = 60):
        inp = f"summarize: {text}"
        ids = self.tokenizer.encode(inp, return_tensors='pt').to(self.device)
        output = self.model.generate(ids, max_new_tokens=max_tokens)
        return self.tokenizer.decode(output[0][ids.shape[-1]:], skip_special_tokens=True)
    def extract(self, text: str, keyword: str):
        return [s for s in text.split(".") if keyword in s]