# rag-app
全流程的 RAG demo  
![Alt text](images/img.png)

## 特点  

- 同时支持普通对话和上传文件RAG对话模式
- 支持OpenAI API调用
- 支持 txt, pdf, docx, markdown 文件类型，并针对中文段落和标点符号做了优化
- 支持对话式的RAG，让RAG也有上下文记忆，RAG 对话始终会以最新上传的文件作为知识库资料

## examples
```bash
LLM_API_BASE=http://192.168.0.20:38060/v1 LLM_API_KEY=EMPTY LLM_API_MODEL=Qwen/Qwen2.5-14B-Instruct LLM_API_EMBEDDING=text-embedding LLM_API_RERANK=rerank python3 src/rag_gradio.py
```