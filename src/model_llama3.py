from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
from typing import List, Optional, Any

def load_llama3(model_path, device):
    model_config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=model_config,
        torch_dtype=torch.bfloat16, device_map=device)
    return model, tokenizer


def llama3_stream_chat(query, history, model, tokenizer, **generate_kwargs: Any):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    # prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # print(f"{prompt}\n\n")
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    generation_kwargs = dict(
        inputs=input_ids,
        streamer=streamer,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        **generate_kwargs,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer
