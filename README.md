# Docuchat

**By: Nicolas Rojas**

Documentation-based RAG pipeline.

This repository contains the source code to build a RAG application that allows questioning a chatbot about internal documentation, using LlamaIndex and Ollama with locally-deployed large language models.

## Installation

Run the following command to install the dependencies:

```shell
sh setup.sh
```

This will install the required python libraries, the language model, and create the documentation database.

## Running

The RAG backend can be run with the following command:

```shell
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

This will mount an API on `localhost:8000`, that can be accessed with the frontend interface.

To use the frontend web application, run `python frontend.py` and the interface will open on `localhost:7860`. You will be able to ask questions about the documents from there.
