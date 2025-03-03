import sys, os, re, copy
from typing import Tuple, List
import gradio as gr
import html

sys.path.append(f"{os.path.dirname(__file__)}/..")

from logger import logger
from model_openai_api import load_openai_api, openai_api_stream_chat
import config
from knowledge_base import (
    init_embeddings, init_reranker,
    check_kb_exist, list_kb_keys,
    embedding_query, rerank_documents, load_documents,
    split_documents, embedding_documents, 
)

def generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k) -> Tuple[str, List, List]:
    query = chat_history[-1]["content"]
    # logger.info(f"KB query: {query}")
    new_query = regenerate_question(chat_history)
    searched_docs = embedding_query(new_query, kb_file, embedding_top_k)
    rerank_docs = rerank_documents(new_query, searched_docs, rerank_top_k)

    if len(rerank_docs) == 0:
        logger.info(f"KB matched empty documents with: {kb_file}")
        return query, [], []
    
    knowledge = ""
    for idx, document in enumerate(rerank_docs):
        knowledge = f"{knowledge}\n\n{document.page_content}"
    knowledge = knowledge.strip()
    # logger.info(f"KB knowledge:\n{knowledge}")

    system_prompt = f"请使用提供的已知信息来回答问题。如果你不知道答案，就说不知道。请保持答案简洁和准确，不要输出与问题不相关的内容。\n以下是已知信息:\n\n{knowledge}"
    history = build_kb_history(chat_history, system_prompt)
    return query, rerank_docs, history


def regenerate_question(chat_history):
    system_prompt = "以下是一段聊天记录，请将该聊天记录的最后一个问题，重新表述为一个独立的问题，使得该问题无需整个聊天记录的上下文即可被理解。不要回答此问题，只需重新表述此问题，如果无法表述就按原样返回。"
    
    query = chat_history[-1]["content"]
    history = build_kb_history(chat_history, "")
    if len(history) <= 1:
        return query
    
    history.append({"role": "user", "content": query})
    prompt = f"{system_prompt}\n聊天记录如下:\n"
    for his in history:
        role = his["role"]
        content = his["content"]
        if role == "assistant":
            prompt += f"答: {content.strip()}\n"
        if role == "user":
            prompt += f"问: {content.strip()}\n"
    # print(f"regenerate_question: {prompt}")
    new_query = ""
    for new_token in chat_streamer(prompt, [], 0.1):
        new_query += new_token
    think_pattern = r"<think>.*?</think>"
    new_query = re.sub(think_pattern, "", new_query, flags=re.DOTALL)
    new_query = new_query.strip()
    logger.info(f"KB new query: {new_query}")
    return new_query


def build_kb_history(chat_history, system_prompt):
    history = chat_history[:-1]

    kb_index = None
    for index in range(len(history)-1, -1, -1):
        item = history[index]
        if isinstance(item["content"], tuple):
            kb_index = index
            break
    if kb_index is not None:
        if kb_index + 1 < len(history):
            history =  history[index + 1:]
        else:
            history = []
    else:
        raise RuntimeError(f"KB can not extract history that start from latest knowledge")
    
    # history = [item for item in history if not isinstance(item["content"], tuple)]
    history = history[-config.history_dialog_limit:]
    history = copy.deepcopy(history)
    details_pattern = r"<details>.*?</details>"
    # think_pattern = r"<think>.*?</think>"
    for item in history:
        del item["metadata"]
        if item["content"] != "":
            item["content"] = re.sub(details_pattern, "", item["content"], flags=re.DOTALL)
            # item["content"] = re.sub(think_pattern, "", item["content"], flags=re.DOTALL)
    if system_prompt and system_prompt.strip() != "":
        history.insert(0, {
            "role": "system",
            "content": system_prompt,
        })
    return history


def generate_chat_prompt(chat_history) -> Tuple[str, List]:
    history = chat_history[:-1]
    history = history[-config.history_dialog_limit:]
    history = copy.deepcopy(history)
    for item in history:
        del item["metadata"]
    query = chat_history[-1]["content"]
    return query, [], history


def generate_query(chat_history, kb_file, embedding_top_k, rerank_top_k):
    if kb_file is None:
        query, searched_docs, history = generate_chat_prompt(chat_history)
    else:
        query, searched_docs, history = generate_kb_prompt(
            chat_history, kb_file, embedding_top_k, rerank_top_k)
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


def handle_chat(chat_history, temperature,
    embedding_top_k=config.embedding_top_k, rerank_top_k=config.rerank_top_k):
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

    generated_text = ""
    for chunk in chat_streamer(query, history, temperature):
        # print(f"chunk: {chunk}")
        generated_text += chunk
        generated_text = generated_text.replace("<think>", "<details><summary>思考过程</summary><div>")
        generated_text = generated_text.replace("</think>", "</div></details>")
        yield chat_resp(chat_history, generated_text)
    print(generated_text)
    if len(searched_docs) > 0:
        knowledge = ""
        for idx, doc in enumerate(searched_docs):
            knowledge = f"{knowledge}{str(doc.page_content).strip()}\n\n"
        knowledge = knowledge.strip()
        knowledge = knowledge.replace("\n", "<br>")
        generated_text += f"<details><summary>参考信息</summary>{knowledge}</details>"
        yield chat_resp(chat_history, generated_text)
    # print(f"chat_history:\n\n{chat_history}\n\n\n\n")


def init_llm():
    load_openai_api()


def chat_streamer(query, history, temperature):
    for chunk in openai_api_stream_chat(query, history, temperature=temperature):
        yield chunk


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
    err = handle_upload_file(file_path, chunk_size=config.chunk_size)
    if err:
        return "", err
    return file_path, None


def handle_upload_file(upload_file: str, chunk_size: int):
    logger.info(f"Handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)

    documents = load_documents(upload_file)
    if isinstance(documents, str):
        err = documents
        logger.error(f"Load documents: {err}")
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
                    f"- 模型: {config.api_model}  \n"
                    # f"- embeddings: {embedding_model_name}  \n"
                    # f"- rerank: {rerank_model_name}  \n"
                    f"- 支持 txt, pdf, docx, markdown")
                temperature = gr.Number(value=0.1, minimum=0.01, maximum=10.00, label="temperature")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="chat", show_label=False, type="messages", min_height=550)
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
    app.queue(max_size=10).launch(server_name='0.0.0.0',
        server_port=config.server_port, show_api=False,
        share=False, favicon_path="icons/shiba.svg")
