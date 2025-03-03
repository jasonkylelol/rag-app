import sys, os
import re, math
import requests, json
from yarl import URL
import httpx
from pydantic import BaseModel

from logger import logger
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from custom.text_splitter import ChineseRecursiveTextSplitter
from custom.text_splitter.markdown_splitter import split_markdown_documents, load_markdown
from langchain_openai.embeddings.base import OpenAIEmbeddings

from config import (
    api_base, api_key, api_embedding, api_rerank, embedding_score_threshold, rerank_top_k
)

vector_db_dict = {}
embedding_model = None

class RerankDocument(BaseModel):
    index: int
    text: str
    score: float

def check_kb_exist(kb_file):
    return True if kb_file in vector_db_dict.keys() else False


def list_kb_keys():
    return vector_db_dict.keys()


def load_documents(upload_file: str):
    file_basename = os.path.basename(upload_file)
    basename, ext = os.path.splitext(file_basename)
    if ext == '.pdf':
        loader = RapidOCRPDFLoader(upload_file)
        documents = loader.load()
    elif ext == '.docx':
        loader = RapidOCRDocLoader(upload_file)
        documents = loader.load()
    elif ext == '.txt':
        loader = UnstructuredLoader(upload_file, autodetect_encoding=True)
        documents = loader.load()
    elif ext == '.md':
        documents = load_markdown(upload_file)
        return documents
    else:
        return "支持 txt pdf docx markdown 文件"
    doc_meta = None
    doc_page_content = ""
    for idx, doc in enumerate(documents):
        if idx == 0:
            doc_meta = doc.metadata
        cleaned_page_content = re.sub(r'\n+', ' ', doc.page_content)
        doc_page_content = f"{doc_page_content}\n{cleaned_page_content}"
    documents = [Document(page_content=doc_page_content, metadata=doc_meta)]
    return documents


def split_documents(file_basename, documents: list, chunk_size: int):
    basename, ext = os.path.splitext(file_basename)
    chunk_overlap = int(chunk_size / 4)
    if ext == '.md':
        documents = split_markdown_documents(documents, chunk_size)
    else:
        documents = split_text_documents(documents, chunk_size, chunk_overlap)
    logger.info(f"file: {file_basename} split to {len(documents)} chunks")
    return documents


def embedding_documents(upload_file, documents):
    global vector_db_dict
    vector_db = FAISS.from_documents(documents, embedding_model,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        relevance_score_fn=custom_relevance_score_fn)
    # file_basename = os.path.basename(upload_file)
    # vector_db_key = f"{file_basename}({human_readable_size(upload_file)})"
    vector_db_key = upload_file
    if vector_db_key in vector_db_dict.keys():
        del vector_db_dict[vector_db_key]
    vector_db_dict[vector_db_key] = vector_db
    return vector_db_key


def split_text_documents(documents: list, chunk_size, chunk_overlap: int):
    if chunk_size > 300:
        full_docs = []
        all_chunk_size = [chunk_size-100, chunk_size, chunk_size+100]
        for auto_chunk_size in all_chunk_size:
            auto_chunk_overlap = int(auto_chunk_size / 4)
            logger.info(f"[split_documents] auto_chunk_size:{auto_chunk_size} auto_chunk_overlap:{auto_chunk_overlap}")
            text_splitter = ChineseRecursiveTextSplitter(
                chunk_size=auto_chunk_size,
                chunk_overlap=auto_chunk_overlap,
            )
            docs = text_splitter.split_documents(documents)
            full_docs.extend(docs)
        return full_docs

    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def embedding_query(query, kb_file, embedding_top_k):
    vector_db = vector_db_dict.get(kb_file)

    # searched_docs = vector_db.similarity_search(query, k=embedding_top_k)
    searched_docs = vector_db.similarity_search_with_relevance_scores(query, k=embedding_top_k)
    # embedding_vectors = embedding_model.embed_query(query)
    # searched_docs = vector_db.similarity_search_by_vector(embedding_vectors, k=embedding_top_k)
    # searched_docs = vector_db.similarity_search_with_score_by_vector(embedding_vectors, k=embedding_top_k)
    docs = []
    for searched_doc in searched_docs:
        doc = searched_doc[0]
        score = searched_doc[1]
        # print(f"{score} : {doc.page_content}")
        if score < embedding_score_threshold:
            continue
        docs.append(doc)
    logger.info(f"[embedding] fetched {len(docs)} docs")
    return docs


def rerank_documents(query, docs, rerank_top_k):
    if len(docs) < 2:
        return docs
    if not api_rerank:
        return docs[:rerank_top_k]
    
    doc_contents = []
    for doc in docs:
        doc_contents.append(doc.page_content)
    
    server_url = api_base
    model_name = api_rerank

    if not server_url:
        raise RuntimeError("server_url is required")
    if not model_name:
        raise RuntimeError("model_name is required")

    url = server_url
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": model_name,
        "query": query,
        "documents": doc_contents,
        "top_n": rerank_top_k,
        "return_documents": True
    }

    try:
        response = requests.post(str(URL(url) / "rerank"), headers=headers, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        results = response.json()
        rerank_documents = []
        for result in results["results"]:
            document = result.get("document", {})
            if document:
                if isinstance(document, dict):
                    text = document.get("text")
                elif isinstance(document, str):
                    text = document
            rerank_document = RerankDocument(
                index=result["index"],
                text=text,
                score=result["relevance_score"],
            )
            rerank_documents.append(rerank_document)

        rerank_docs = []
        for rerank_document in rerank_documents:
            rerank_docs.append(Document(page_content=rerank_document.text))
        
        logger.info(f"[rerank] fetched {len(rerank_docs)} docs")
        return rerank_docs

    except httpx.HTTPStatusError as e:
        raise RuntimeError(str(e))


def init_embeddings():
    global embedding_model

    logger.info(f"Using embedding: {api_embedding}")
    embedding_model = OpenAIEmbeddings(
        model = api_embedding,
        base_url = api_base,
        api_key= api_key,
        check_embedding_ctx_length = False,
    )


def init_reranker():
    logger.info(f"Using reranker: {api_rerank}")


def custom_relevance_score_fn(distance: float) -> float:
    score = 1.0 - distance / math.sqrt(2)
    score = 0 if score < 0 else score 
    return score


def human_readable_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_names = ('KB', 'MB', 'GB')
    i = 0
    while size_bytes >= 1024 and i < len(size_names):
        size_bytes /= 1024.0
        i += 1
    return '{:.2f} {}'.format(size_bytes, size_names[i-1])
