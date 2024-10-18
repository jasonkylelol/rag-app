# config

import os

server_port = 8060

device="cuda"

model_path = "/root/huggingface/models"
# model_name = "THUDM/glm-4-9b-chat"
# model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model_name = "glm-4-plus"
model_full = f"{model_path}/{model_name}"

api_key = os.getenv("LLM_RAG_API_KEY")

embedding_model_name = "maidalun1020/bce-embedding-base_v1"
embedding_model_full = f"{model_path}/{embedding_model_name}"

rerank_model_name = "maidalun1020/bce-reranker-base_v1"
rerank_model_full = f"{model_path}/{rerank_model_name}"

max_new_tokens=8192
top_p=0.1
# temperature=0.1
chunk_size=300
embedding_top_k=10
rerank_top_k=3
history_dialog_limit=10

embedding_score_threshold = 0.1
