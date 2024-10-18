from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
)
import torch
from typing import List, Optional, Any
from threading import Thread
from logger import logger

def load_glm4(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, encode_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device,
    ).eval()
    return model, tokenizer


def glm4_stream_chat(query, history, model, tokenizer, **generate_kwargs: Any):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    logger.info(f"Chat messages:\n{messages}")
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = model.config.eos_token_id
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    stop = StopOnTokens()
    generate_params = dict(
        input_ids=inputs,
        streamer=streamer,
        # max_new_tokens=max_length,
        do_sample=True,
        # top_p=top_p,
        # temperature=temperature,
        stopping_criteria=StoppingCriteriaList([stop]),
        repetition_penalty=1.2,
        eos_token_id=model.config.eos_token_id,
        **generate_kwargs,
    )
    # logger.info(f"generate_kwargs: {generate_params}")
    thread = Thread(target=model.generate, kwargs=generate_params)
    thread.start()
    return streamer
