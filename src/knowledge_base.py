import sys, os
import re, math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from logger import logger
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from custom.document_loaders import RapidOCRPDFLoader, RapidOCRDocLoader
from custom.text_splitter import ChineseRecursiveTextSplitter
from custom.text_splitter.markdown_splitter import split_markdown_documents, load_markdown
import config

embedding_model = None
rerank_model, rerank_tokenizer = None, None

vector_db_dict = {}

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
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE, relevance_score_fn=custom_relevance_score_fn)
    file_basename = os.path.basename(upload_file)
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
        if score < config.embedding_score_threshold:
            continue
        docs.append(doc)
    logger.info(f"[embedding] fetched {len(docs)} docs")
    return docs


def rerank_documents(query, docs, rerank_top_k):
    if len(docs) < 2:
        return docs
    if not rerank_model:
        return docs[:rerank_top_k]

    pairs = []
    for idx, document in enumerate(docs):
        pairs.append([query, document.page_content])
    rerank_docs = []
    with torch.no_grad():
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(config.device)
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = torch.sigmoid(scores)
        scores = scores.tolist()
    
    # print(f"scores: {scores}")
    combined_list = list(zip(docs, scores))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
    for idx, item in enumerate(sorted_combined_list):
        if idx >= rerank_top_k:
            break
        document = item[0]
        rerank_docs.append(document)
    logger.info(f"[rerank] fetched {len(rerank_docs)} docs")
    return rerank_docs


def init_embeddings():
    global embedding_model

    logger.info(f"Load embedding from {config.embedding_model_full}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.embedding_model_full,
        model_kwargs={"device": config.device},
        encode_kwargs={'normalize_embeddings': True},
    )


def init_reranker():
    global rerank_model, rerank_tokenizer

    logger.info(f"Load reranker from {config.rerank_model_full}")

    if config.rerank_model_full:
        rerank_tokenizer = AutoTokenizer.from_pretrained(
            config.rerank_model_full,
            device_map=config.device)
        rerank_model = AutoModelForSequenceClassification.from_pretrained(
            config.rerank_model_full,
            device_map=config.device)
        rerank_model = rerank_model.eval()


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
