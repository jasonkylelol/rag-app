import os
from openai import OpenAI
from typing import List, Optional, Any
from logger import logger
from config import api_key, api_model

client, model = None, None

def load_glm4_api():
    global client, model
    client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    model = api_model
    logger.info(f"Using api model: {api_model}")


def glm4_api_stream_chat(query, history, **generate_kwargs: Any):
    messages = []
    if len(history) > 0:
        messages.extend(history)
    messages.append({"role":"user","content":query})
    temperature = generate_kwargs.get("temperature", 0.1)
    logger.info(f"Using model: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for idx, chunk in enumerate(response):
        # print(f"Chunk received, value: {chunk}")
        chunk_message = chunk.choices[0].delta
        if not chunk_message.content:
            continue
        chunk_content = chunk_message.content

        yield chunk_content

        finish_reason = chunk.choices[0].finish_reason
        if finish_reason:
            logger.info(f"\nfinish_reason: {finish_reason}")
