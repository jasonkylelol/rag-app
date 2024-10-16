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
    embedding_top_k, rerank_top_k, server_port
)
from knowledge_base import (
    init_embeddings, init_reranker,
    check_kb_exist, list_kb_keys,
    embedding_query, rerank_documents, load_documents,
    split_documents, embedding_documents, 
)

model, tokenizer = None, None
uploading_status = {}

def generate_kb_prompt(chat_history, kb_file, embedding_top_k, rerank_top_k) -> Tuple[str, List, List]:
    query = chat_history[-1][0]
    
    searched_docs = embedding_query(query, kb_file, embedding_top_k)

    rerank_docs = rerank_documents(query, searched_docs, rerank_top_k)

    logger.info(f"Query: {query}")
    if len(rerank_docs) == 0:
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
    history = chat_history[:-1]
    query = chat_history[-1][0]

    model_history = []
    for hist in history:
        user_msg = hist[0]
        assistant_msg = hist[1]
        if isinstance(user_msg, str):
            model_history.append({"role":"user","content":user_msg})
            logger.info(f"User: {user_msg}")
        else:
            logger.warning(f"Skip user: {user_msg}")
        if isinstance(assistant_msg, str):
            model_history.append({"role":"assistant","content":assistant_msg})
            logger.info(f"Assistant: {assistant_msg}")
        else:
            logger.warning(f"Skip assistant: {assistant_msg}")
    logger.info(f"Query: {query}")
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
    chat_history[-1][1] = msg
    return chat_history


def handle_chat(chat_history, kb_file, temperature, embedding_top_k=embedding_top_k, rerank_top_k=rerank_top_k):
    if kb_file is not None and not check_kb_exist(kb_file):
        err = f"文件: {kb_file} 不存在"
        logger.error(f"[handle_chat] err: {err}")
        yield chat_resp(chat_history, err)
        return
    
    logger.info(f"Handle chat: temperature: {temperature} "
        f"embedding_top_k: {embedding_top_k} rerank_top_k: {rerank_top_k}")

    query, history, searched_docs = generate_query(chat_history, kb_file, embedding_top_k, rerank_top_k)
    
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


def handle_upload_file(upload_file: str, chunk_size: int):
    global uploading_status
    if not upload_file:
        logger.error("invalid upload_file")
        return

    logger.info(f"Handle file: {upload_file}")
    file_basename = os.path.basename(upload_file)

    uploading_status[file_basename] = "正在加载文件..."
    documents = load_documents(upload_file)
    if isinstance(documents, str):
        logger.error(documents)
        uploading_status[file_basename] = documents
        return

    uploading_status[file_basename] = "正在拆分文件..."
    documents = split_documents(file_basename, documents, chunk_size)

    uploading_status[file_basename] = f"拆分文件为{len(documents)}份，正在向量化..."
    vector_db_key = embedding_documents(upload_file, documents)

    uploading_status[file_basename] = f"拆分文件为{len(documents)}份，向量化成功"
    logger.info(f"[chinese_rec_text_splitter] [{vector_db_key}] "
        f"chunk_size: {chunk_size} chunks: {len(documents)}")


def doc_loaded():
    return gr.Radio(choices=list_kb_keys(), label="可供选择的文件:")


def handle_add_msg(query, chat_history):
    # logger.info(query, chat_history)
    for x in query["files"]:
        chat_history.append(((x,), None))
    if query["text"] is not None:
        chat_history.append((query["text"], None))
    return gr.MultimodalTextbox(value=None, interactive=False), chat_history
    # return gr.Textbox(value=None, interactive=False), chat_history + [[query, None]]


def uploading_stat():
    # logging.info(uploading_files)
    if len(uploading_status) == 0:
        return ""
    status_info = ""
    for k in uploading_status.keys():
        v = uploading_status.get(k)
        status_info = f"{status_info}{k}:&emsp;&emsp;{v}  \n"
    return status_info.rstrip()


def init_blocks():
    with gr.Blocks(title="RAG") as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# RAG 🤖  \n"
                    f"- llm: {model_name}  \n"
                    f"- embeddings: {embedding_model_name}  \n"
                    f"- rerank: {rerank_model_name}  \n"
                    f"- 支持 txt, pdf, docx, markdown")
                # upload_file = gr.File(file_types=[".text"], label="对话文件")
                upload_file = gr.UploadButton("对话文件上传")
                upload_stat = gr.Markdown(value=uploading_stat, every=0.5)
                with gr.Row():
                    chunk_size = gr.Number(value=300, minimum=100, maximum=1000, label="chunk_size")
                    # chunk_overlap = gr.Number(value=50, minimum=10, maximum=500, label="chunk_overlap")
                    temperature = gr.Number(value=0.1, minimum=0.01, maximum=0.99, label="temperature")
                # with gr.Row():
                    # embedding_top_k = gr.Number(value=15, minimum=5, maximum=100, label="embedding_top_k")
                    # rerank_top_k = gr.Number(value=3, minimum=1, maximum=5, label="rerank_top_k")
            with gr.Column(scale=5):
                query_doc = doc_loaded()
                chatbot = gr.Chatbot(label="chat", show_label=False)
                with gr.Row():
                    query = gr.MultimodalTextbox(label="chat with picture", show_label=False, scale=4)
                    clear = gr.ClearButton(value="清空聊天记录", components=[query, chatbot], scale=1)
        
        query.submit(
            handle_add_msg, inputs=[query, chatbot], outputs=[query, chatbot]).then(
            handle_chat, inputs=[chatbot, query_doc, temperature], outputs=[chatbot]).then(
            lambda: gr.MultimodalTextbox(interactive=True), outputs=[query])
        upload_file.upload(
            handle_upload_file, inputs=[upload_file, chunk_size], show_progress="full").then(
            doc_loaded, outputs=query_doc)
        app.load(doc_loaded, outputs=query_doc)

    return app


# nohup langchain_demo/rag/svc_start.sh > logs.txt 2>&1 &
if __name__ == "__main__":
    init_embeddings()
    init_reranker()
    init_llm()

    app = init_blocks()
    app.queue(max_size=10).launch(server_name='0.0.0.0', server_port=server_port, show_api=False,
        share=False, favicon_path="/root/github/llm-demo/icons/shiba.svg")
