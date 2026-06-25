import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import json

load_dotenv()

EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    model=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
)

def build_and_save_index(pdf_path: str, index_path: str):
    """Build FAISS index from PDF and save to disk."""
    path = Path(pdf_path)
    if not path.exists():
        print(f"PDF not found: {pdf_path}")
        return
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
    store.save_local(index_path)
    print(f"Index saved to {index_path}")

def store_memory_record(record: dict, index_path: str) -> None:
    """Append a structured memory record (converted to JSON) to the memory index."""

    text = json.dumps(record, ensure_ascii=False)
    doc = Document(page_content=text)
    if Path(index_path).exists():
        try:
            store = FAISS.load_local(index_path, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"store_memory_record: corrupt index, recreating. {e}")
            store = FAISS.from_documents([doc], EMBEDDING_MODEL)
            store.save_local(index_path)
            return
        store.add_documents([doc])
        store.save_local(index_path)
    else:
        store = FAISS.from_documents([doc], EMBEDDING_MODEL)
        store.save_local(index_path)
    print("store_memory_record: memory ingested.")

def similarity_search(index_path: str, query: str, k: int = 3) -> List[str]:
    """Similarity search against given FAISS index path."""
    if not Path(index_path).exists():
        return []
    store = FAISS.load_local(index_path, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    results = store.similarity_search(query, k=k)
    return [r.page_content for r in results]

if __name__ == "__main__":
    # build index from PDF
    build_and_save_index("policy.pdf", 'index.faiss')
