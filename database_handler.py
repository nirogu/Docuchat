"""
CLI handler to create or update the Chroma database.

Author
------
Nicolas Rojas
"""

# import libraries
from argparse import ArgumentParser
import os.path
from datetime import datetime
import yaml
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def create_index(
    chroma_collection_name: str,
    documents_dir: str,
    index_dir: str,
    embedding_model: str,
):
    """Create vector database from documents folder.

    Parameters
    ----------
    chroma_collection_name : str
        Name of the Chroma collection to be created.
    documents_dir : str
        Directory where the documents are stored.
    index_dir : str
        Directory where the index is saved.
    embedding_model : str
        Huggingface embedding model to vectorize the documents.
    """
    # create Chroma vector store
    chroma_client = chromadb.PersistentClient(path=index_dir)
    chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load the documents and create the index
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )


def update_index(
    chroma_collection_name: str,
    documents_dir: str,
    index_dir: str,
    embedding_model: str,
):
    """Update vector database with new or changed files.

    Parameters
    ----------
    chroma_collection_name : str
        Name of the Chroma collection to be updated.
    documents_dir : str
        Directory where the documents are stored.
    index_dir : str
        Directory where the index is saved.
    embedding_model : str
        Huggingface embedding model to vectorize the documents.
    """
    # load the existing index
    chroma_client = chromadb.PersistentClient(path=index_dir)
    chroma_collection = chroma_client.get_or_create_collection(chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    documents = SimpleDirectoryReader(documents_dir).load_data()

    for doc in documents:
        doc_id = doc.metadata["file_path"]

        # Check if document already exists in the index
        existing_node = index.docstore.document_exists(doc_id)

        if existing_node:
            existing_mtime = datetime.fromisoformat(
                existing_node.metadata["last_modified"]
            )
            current_mtime = datetime.fromtimestamp(os.path.getmtime(doc_id))

            # If the file has been modified, update it
            if current_mtime > existing_mtime:
                index.update_ref_doc(doc_id, doc)
        else:
            # If the document doesn't exist, insert it
            index.insert(doc)
    # Persist changes
    index.storage_context.persist(persist_dir=index_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Utility to manage database index with LlamaIndex and Chroma"
    )
    parser.add_argument(
        "-c", "--create", action="store_true", help="Create the database"
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update the database"
    )
    args = parser.parse_args()
    # load configuration variables
    with open("config.yaml", "r", encoding="utf8") as yfile:
        parameters = yaml.safe_load(yfile)

    index_dir = parameters["index_directory"]
    chunk_size = parameters["chunk_size"]
    embedding_model = parameters["embedding_model"]
    chroma_collection = parameters["chroma_collection"]
    documents_dir = parameters["documents_dir"]

    # Set custom RAG settings
    Settings.chunk_size = chunk_size
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # if both arguments are true or both are false, throw error
    if args.create == args.update:
        raise ValueError(
            "Use the program with argument -c OR -u. Use flag -h for help."
        )

    # create database if doesnt exist yet
    if args.create:
        if os.path.exists(index_dir):
            raise FileExistsError(f"The file {index_dir} already exists")
        create_index(chroma_collection, documents_dir, index_dir, embedding_model)

    # update existing database
    elif args.update:
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"The file {index_dir} does not exist")
        update_index(chroma_collection, documents_dir, index_dir, embedding_model)
