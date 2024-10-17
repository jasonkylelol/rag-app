import sys, os
from typing import Tuple, List
import gradio as gr
import signal

sys.path.append(f"{os.path.dirname(__file__)}/..")

from utils import md5sum_str
from logger import logger
from model_llama3 import load_llama3, llama3_stream_chat
from model_glm4 import load_glm4, glm4_stream_chat
from model_glm4_api import load_glm4_api, glm4_api_stream_chat
from config import (
    device, model_full, model_name, max_new_tokens,
    embedding_model_name, rerank_model_name,
    embedding_top_k, rerank_top_k, server_port, chunk_size
)
from knowledge_base import (
    init_embeddings, init_reranker,
    check_kb_exist, list_kb_keys,
    embedding_query, rerank_documents, load_documents,
    split_documents, embedding_documents, 
)

model, tokenizer = None, None

def generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k) -> Tuple[str, List, List]:
    query = chat_history[-1]["content"]
    
    searched_docs = embedding_query(query, kb_file, embedding_top_k)

    rerank_docs = rerank_documents(query, searched_docs, rerank_top_k)

    logger.info(f"KB Query: {query}")
    if len(rerank_docs) == 0:
        logger.info(f"KB matched empty documents with: {kb_file}")
        return query, [], []
    knowledge = ""
    for idx, document in enumerate(rerank_docs):
        knowledge = f"{knowledge}\n\n{document.page_content}"
    knowledge = knowledge.strip()
    logger.info(f"Knowledge:\n{knowledge}")

    kb_query = ("<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，"
        "请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。</指令>\n"
        f"<已知信息>{knowledge}</已知信息>\n<问题>{query}</问题>")
    # kb_query = ("根据以下背景知识回答问题，回答中不要出现（根据上文，根据背景知识，根据文档）等文案，"
    #     "如果问题与背景知识不相关，或无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n"
    #     f"背景知识: \n\n{knowledge}\n\n问题: {query}")
    return kb_query, rerank_docs, []


def generate_chat_prompt(chat_history) -> Tuple[str, List]:
    model_history = chat_history[:-1]
    query = chat_history[-1]["content"]
    return query, model_history


def generate_query(chat_history, kb_file, embedding_top_k, rerank_top_k):
    if kb_file is None:
        query, history = generate_chat_prompt(chat_history)
        searched_docs = []
        # yield chat_resp(chat_history, "需要选择文件")
        # return
    else:
        query, searched_docs, history = generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k)
    return query, history, searched_docs


def chat_resp(chat_history, msg):
    if chat_history[-1]["role"] == "assistant":
        chat_history[-1]["content"] = msg
    else:
        chat_history.append({
            "role": "assistant",
            "content": msg,
        })
    return chat_history


def handle_chat(chat_history, temperature, embedding_top_k=embedding_top_k, rerank_top_k=rerank_top_k):
    kb_file, err = build_knowledge(chat_history)
    if err:
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return
    
    logger.info(f"Handle chat: kb_file: {kb_file} temperature: {temperature} "
        f"embedding_top_k: {embedding_top_k} rerank_top_k: {rerank_top_k}")
    
    # print(f"chat_history:\n\n{chat_history}\n")

    query, history, searched_docs = generate_query(chat_history, kb_file, embedding_top_k, rerank_top_k)
    if query.strip() == "":
        err = "prompt can not be empty"
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return
    
    if "glm-4-api" == model_name:
        streamer = glm4_api_stream_chat(query, history, temperature=temperature)
    elif "glm-4" in model_name.lower():
        streamer = glm4_stream_chat(query, history, model, tokenizer,
            temperature=temperature, max_new_tokens=max_new_tokens)
    elif "llama3" in model_name.lower():
        streamer = llama3_stream_chat(query, history, model, tokenizer,
            temperature=temperature, max_new_tokens=max_new_tokens)
    else:
        raise RuntimeError(f"f{model_name} is not support")
    
    generated_text = ""
    for new_token in streamer:
        generated_text += new_token
        yield chat_resp(chat_history, generated_text)
    if len(searched_docs) > 0:
        knowledge = ""
        for idx, doc in enumerate(searched_docs):
            knowledge = f"{knowledge}{doc.page_content}\n\n"
        knowledge = knowledge.strip()
        generated_text += f"<details><summary>参考信息</summary>{knowledge}</details>"
        yield chat_resp(chat_history, generated_text)

    # print(f"chat_history:\n\n{chat_history}\n\n\n\n")


def init_llm():
    global model, tokenizer

    logger.info(f"Load from {model_name}")
    if "glm-4-api" == model_name:
        load_glm4_api()
    elif "glm-4" in model_name.lower():
        model, tokenizer = load_glm4(model_full, device)
    elif "llama3" in model_name.lower():
        model, tokenizer = load_llama3(model_full, device)
    else:
        raise RuntimeError(f"{model_name} is not support")


def build_knowledge(chat_history):
    file_path = None
    for item in reversed(chat_history):
        if item["role"] == "user" and isinstance(item["content"], tuple):
            for path in item["content"]:
                file_path = path
            break
    if file_path is None:
        logger.info(f"Knowledge not found in chat_history")
        return None, None
    if check_kb_exist(file_path):
        logger.info(f"Knowledge {file_path} found in existing data")
        return file_path, None
    err = handle_upload_file(file_path, chunk_size=chunk_size)
    if err:
        return "", err
    return file_path, None


def handle_upload_file(upload_file: str, chunk_size: int):
    logger.info(f"Handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)

    documents = load_documents(upload_file)
    if isinstance(documents, str):
        err = documents
        logger.error(err)
        return err

    logger.info(f"Splitting file: {upload_file} ...")
    documents = split_documents(file_basename, documents, chunk_size)

    logger.info(f"Split file to {len(documents)} chunks, embedding...")
    embedding_documents(upload_file, documents)

    logger.info(f"Embedding succeed, chunk_size: {chunk_size}")


def handle_add_msg(query, chat_history):
    # print(f"query:\n\n{query}\n")
    # print(f"chat_history:\n\n{chat_history}\n")
    for x in query["files"]:
        chat_history.append({
            "role": "user",
            "content": {"path": x},
        })
    if query["text"] is not None:
        chat_history.append({
            "role": "user",
            "content": query["text"],
        })
    return gr.MultimodalTextbox(value=None, interactive=False), chat_history


def init_blocks():
    with gr.Blocks(title="RAG") as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# RAG 🤖  \n"
                    f"- llm: {model_name}  \n"
                    f"- embeddings: {embedding_model_name}  \n"
                    f"- rerank: {rerank_model_name}  \n"
                    f"- 支持 txt, pdf, docx, markdown")
                temperature = gr.Number(value=0.1, minimum=0.01, maximum=0.99, label="temperature")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="chat", show_label=False, type="messages", min_height=600)
                with gr.Row():
                    query = gr.MultimodalTextbox(label="chat", show_label=False, scale=4)
                    gr.ClearButton(value="清空聊天记录", components=[query, chatbot], scale=1)
        
        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
            handle_chat, inputs=[chatbot, temperature], outputs=[chatbot]).then(
            lambda: gr.MultimodalTextbox(interactive=True), outputs=[query])

    return app


if __name__ == "__main__":
    init_embeddings()
    init_reranker()
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=server_port, show_api=False,
        share=False, favicon_path="icons/shiba.svg")
