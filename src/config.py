# config

import os

server_port = 8061

api_base = os.getenv("LLM_API_BASE")
api_key = os.getenv("LLM_API_KEY")
api_model = os.getenv("LLM_API_MODEL")

model_path = "/root/huggingface/models"
device = "cuda"

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
