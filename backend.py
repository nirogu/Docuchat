"""
RAG pipeline accessible through RestAPI.

Author
------
Nicolas Rojas
"""

# import libraries
import os.path
import yaml
from json import dumps
from pydantic import BaseModel
from fastapi import FastAPI
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from database_handler import create_index


class Query(BaseModel):
    query: str


def save_query(path: str, query: str, response: dict):
    """Save query in persistent jsonl file.

    Parameters
    ----------
    path : str
        Path to the jsonl file where the query history is saved.
    query : str
        Query received by the RAG system.
    response : dict
        Dictionary with the response and relevant documents.
    """
    data = dict(response)
    data["query"] = query
    with open(path, "a", encoding="utf8") as jfile:
        jfile.write(dumps(data, ensure_ascii=False) + "\n")


# load configuration variables
with open("config.yaml", "r", encoding="utf8") as yfile:
    parameters = yaml.safe_load(yfile)

index_dir = parameters["index_directory"]
chunk_size = parameters["chunk_size"]
embedding_model = parameters["embedding_model"]
ollama_model = parameters["ollama_model"]
chroma_collection = parameters["chroma_collection"]
documents_dir = parameters["documents_dir"]
query_history = parameters["query_history"]

# Set custom RAG settings
Settings.chunk_size = chunk_size
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
Settings.llm = Ollama(model=ollama_model, request_timeout=360.0)

# initiate FastAPI app
app = FastAPI()

# check if stored index already exists
if not os.path.exists(index_dir):
    create_index(chroma_collection, documents_dir, index_dir, embedding_model)

# load the existing index
chroma_client = chromadb.PersistentClient(path=index_dir)
chroma_collection = chroma_client.get_or_create_collection(chroma_collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbedding(model_name=embedding_model)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# define the index
query_engine = index.as_query_engine()


@app.post("/query/")
def retrieve(query: Query) -> dict:
    """Run a query with the RAG pipeline.

    Parameters
    ----------
    query : str
        Question asked by the user, as a string.

    Returns
    -------
    dict
        Dictionary containing the answer given by the LLM and the relevant
        documents.
    """
    query = query.query
    global query_engine
    response = query_engine.query(query)
    result = {"response": response.response}

    source_files = []
    for source_node in response.source_nodes:
        source_files.append(source_node.node.metadata["file_name"])
    source_files = list(set(source_files))
    result["source_files"] = source_files

    save_query(query_history, query, result)
    return result
