from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from custom.text_splitter import ChineseRecursiveTextSplitter
from langchain_core.documents import Document
import re


custom_separators=[
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "!",
    "?",
    ";",
    "\u200B",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
    "。",
    "，",
    "；",
    "！",
    "？",
]


def print_docs(documents):
    for document in documents:
        # print(f"{document.metadata}\t{document.page_content}\n\n")
        print(f"{document.page_content}\n\n---------------------------------------------------------------\n")


def load_markdown(file):
    with open(file, "r") as f:
        content = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = splitter.split_text(content)
    return documents


def split_markdown_documents(documents, chunk_size):
    new_documents = []
    for document in documents:
        header = ""
        for k in document.metadata:
            header = f"{header}\n{document.metadata.get(k)}"
        header = header.strip()

        document.page_content = re.sub(r'\n+', ' ', document.page_content)
        html_contents, cleaned_content = extract_table(document.page_content)
        
        new_documents.extend(split_by_recursive(header, cleaned_content, chunk_size))

        for html_content in html_contents:
            html_content = f"{header}\n{html_content}"
            new_documents.append(Document(page_content=html_content))

    return new_documents


def split_by_recursive(header, cleaned_content, chunk_size):
    documents = [Document(page_content=cleaned_content)]
    text_splitter = ChineseRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 4),
        )
    # text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=chunk_size,
    #             chunk_overlap=int(chunk_size / 4),
    #             separators=custom_separators,
    #         )
    documents = text_splitter.split_documents(documents)
    for document in documents:
        document.page_content = f"{header}\n{document.page_content}"
    return documents


def extract_table(html_string):
    html_contents = re.findall(r'<table>(.*?)</table>', html_string, re.DOTALL)
    cleaned_content = re.sub(r'<table>(.*?)</table>', '', html_string, flags=re.DOTALL)
    html_contents = ['<table>' + item + '</table>' for item in html_contents]
    return html_contents, cleaned_content


if __name__ == "__main__":
    # file = "cache/files/平台对接说明.pdf_ocr.md"
    file = "cache/files/科大讯飞2023半年报.pdf_ocr.md"

    documents = load_markdown(file)
    documents = split_markdown_documents(documents, chunk_size=300)
    print_docs(documents)
