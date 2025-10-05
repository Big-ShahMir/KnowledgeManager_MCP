# Modern LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings # Don't want to run models locally
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings



# Alternative/additional options
import openai
from pypdf import PdfMerger  # More modern than PyPDF2
from pydantic.v1 import SecretStr
import os           
from dotenv import load_dotenv      # Reading .env files
import asyncio
import glob
import re
import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Any

# MCP imports
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize Fast MCP
mcp = FastMCP("KnowledgeManager")

# Load environment variables from .env file
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" # HuggingFace text embedding model

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in .env file or environment variables")

# initializing text embeddings model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=SecretStr(HUGGINGFACE_API_KEY),
    model_name="BAAI/bge-large-en-v1.5"
)

# Global variable to hold the vector store in memory
vector_store: FAISS | None = None
DB_FILE = "logs.db"

def init_db():
    """Initializes the SQLite database and creates the logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            inputs TEXT NOT NULL,
            output TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_usage(tool_name: str, inputs: dict, output: str):
    """Logs a tool usage event to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usage_logs (timestamp, tool_name, inputs, output) VALUES (?, ?, ?, ?)",
                   (datetime.utcnow().isoformat(), tool_name, json.dumps(inputs), output))
    conn.commit()
    conn.close()

# Helper functions ===============================================

# Async function to load and split PDF documents
async def load_and_split_pdf(file_path: str) -> list[Document]:
    """Load and split a PDF file into text chunks."""
    loader = PyPDFLoader(file_path)
    # Use the async version of the loader
    documents = await loader.aload()

    # Snip the path and keep only the filename for the 'source' metadata
    file_name = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = file_name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    return texts
 
# Synchronous version of load_and_split_pdf
# def load_and_split_pdf(file_path: str) -> list[str]:
#     """Load and split a PDF file into text chunks."""
#     loader = PyPDFLoader(file_path)
#     # Use the sync version of the loader
#     documents =  loader.aload()

#     # Snip the path and keep only the filename for the 'source' metadata
#     file_name = os.path.basename(file_path)
#     for doc in documents:
#         doc.metadata["source"] = file_name

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP
#     )
#     texts = text_splitter.split_documents(documents)
#     return texts


def create_vector_store(texts: list[Document]) -> FAISS:
    """Create a FAISS vector store from text chunks."""
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def filter_results (query: str, res: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """Filter results based on query terms."""
    q_length = len(query)
    
    score_threshold = None
    if q_length <= 4:      # Simple queries: stricter
        score_threshold = 0.4
    elif q_length <= 10:    # Medium queries: balanced  
        score_threshold = 0.6
    else:                      # Complex queries: permissive
        score_threshold = 0.8

    filtered = [(doc, score) for doc, score in res if score < score_threshold]
    return filtered if filtered and len(filtered) > 1 else res

# MCP Tools ======================================================

# @mcp.tool
# async def add_document_to_knowledge_base(file_path: str) -> str:
#     """
#     Loads a PDF from a file path, splits it into chunks,
#     and creates an in-memory vector store for querying.
#     """
#     global vector_store
#     try:
#         print(f"Loading and processing file: {file_path}")
#         texts = await load_and_split_pdf(file_path)
#         print(f"Creating vector store from {len(texts)} text chunks...")
#         # Use the async method to create the vector store
#         vector_store = await FAISS.afrom_documents(texts, embeddings)
#         return f"Successfully created knowledge base from {file_path} with {len(texts)} chunks."
#     except Exception as e:
#         return f"An error occurred: {e}"

@mcp.tool()
async def query_knowledge_base(query: str) -> str:
    """Queries the in-memory knowledge base to find relevant information."""
    if vector_store is None:
        return "Error: The knowledge base has not been created yet. Please use 'add_document_to_knowledge_base' first."

    # Use the async version to avoid blocking the event loop
    docs = await vector_store.asimilarity_search_with_score(query, k=7)

    # filter results based on score thresholds
    results = filter_results(query, docs)

    # Format the results for a clean output
    results = [f"Source: {doc[0].metadata.get('source', 'Unknown')}, Page: {doc[0].metadata.get('page', 'N/A')}:\n{doc[0].page_content}\n" for doc in results]
    output_str = "\n---\n".join(results) if results else "No relevant documents found."
    
    log_usage("query_knowledge_base", {"query": query}, output_str)
    return output_str


@mcp.tool()
async def search_specific_documents(query: str, document_names: list[str]) -> str:
    """
    An MCP tool to answer questions using relevant sources from vector DB
    
    Args:
        question: Natural language question (e.g., "What day is Steve Jobs birthday from his biography?")
        max_sources: Maximum number of source chunks to retrieve
    
    Returns:
        Dict with answer, sources, and confidence info
    """
    if vector_store is None:
        return "Error: The knowledge base has not been created yet. Please use 'add_document_to_knowledge_base' first."
    
    max_results = len(document_names) * 7  # Number of documents to retrieve

    # Use the async version to avoid blocking the event loop
    results = await vector_store.asimilarity_search_with_score(query, k=max_results)
    
    filtered_results = []
    for doc, score in results:
        source = doc.metadata.get('source', '').lower()
        if any(doc_name.lower() in source for doc_name in document_names):
            # filtered_results.append({
            #     'content': doc.page_content,
            #     'source': doc.metadata.get('source'),
            #     'score': score
            # })
            filtered_results.append((doc, score))
    
    final_filter = filter_results(query, filtered_results)
    
    
    # Format the results for a clean, string-based output
    results_str = [f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}\n" for doc, score in final_filter]
    output_str = "\n---\n".join(results_str) if results_str else "No relevant documents found in the specified sources."

    log_usage("search_specific_documents", {"query": query, "document_names": document_names}, output_str)
    return output_str

    
if __name__ == "__main__":
    async def main():
        """
        Main async function to initialize the vector store and run the MCP server.
        """
        global vector_store

        print("Initializing logging database...")
        init_db()
        
        # Assume PDFs are in a 'data' directory at the project root
        # Project Root -> data/
        #            -> KM/src.py
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, "data")
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

        if not pdf_files:
            print("Warning: No PDF files found in the 'data' directory. The knowledge base will be empty.")
        else:
            print(f"Found {len(pdf_files)} PDF(s) to load into the knowledge base.")
            all_texts = []
            for pdf_path in pdf_files:
                print(f"  - Processing {os.path.basename(pdf_path)}...")
                texts = await load_and_split_pdf(pdf_path)
                all_texts.extend(texts)
            
            vector_store = create_vector_store(all_texts)
            print(f"Knowledge base created successfully with {len(all_texts)} document chunks.")

        print("Starting KnowledgeManager MCP server...")
        mcp.run(transport='stdio')

    asyncio.run(main())