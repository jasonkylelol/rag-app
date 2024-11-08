from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Optional, Any
from threading import Thread
from logger import logger


def load_qwen2(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def qwen2_stream_chat(query, history, model, tokenizer, **generate_kwargs: Any):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    logger.info(f"Chat messages:\n{messages}")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        do_sample=True,
        **generate_kwargs,
    )
    logger.info(f"generate_kwargs: {generate_kwargs}")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer
