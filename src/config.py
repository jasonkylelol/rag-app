# config

import os

server_port = 8061

api_base = os.getenv("LLM_API_BASE")
api_key = os.getenv("LLM_API_KEY")
api_model = os.getenv("LLM_API_MODEL")
api_embedding = os.getenv("LLM_API_EMBEDDING")
api_rerank = os.getenv("LLM_API_RERANK")

max_new_tokens=8192
top_p=0.1
# temperature=0.1
chunk_size=300
embedding_top_k=10
rerank_top_k=3
history_dialog_limit=10

embedding_score_threshold = 0.1
