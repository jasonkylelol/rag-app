import os
from openai import OpenAI
from typing import List, Optional, Any
from logger import logger
from config import api_base, api_key, api_model, max_new_tokens

client = None

def load_openai_api():
    global client
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    logger.info(f"Using api base: {api_base}, model: {api_model}")


def openai_api_stream_chat(query, history, **generate_kwargs: Any):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    temperature = generate_kwargs.get("temperature", 0.1)
    logger.info(f"Chat messages:\n{messages}")
    response = client.chat.completions.create(
        model=api_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        stream=True,
    )

    for idx, chunk in enumerate(response):
        chunk_content = chunk.choices[0].delta.content

        if not chunk_content:
            continue

        yield chunk_content

        finish_reason = chunk.choices[0].finish_reason
        if finish_reason:
            logger.info(f"\nfinish_reason: {finish_reason}")
