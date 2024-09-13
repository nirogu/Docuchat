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

## Rationale

This repository addresses the problem of the significant amount of time that developers spend searching through company documentation, or asking other developers simple questions that are in the documentation. An efficient way of solving this is building an intelligent agent with access to the company's internal documentation, so that the users can asks questions in natural language as they would interact with an experienced human. An adequate LLM pattern that can be used in this situation is called Retrieval-Augmented generation (RAG), which is an application that can answer questions about specific source information. In other words, is a sophisticated question-answering (Q&A) chatbot, that can reason about private data or data introduced after a model's cutoff date.

An important issue is that, sometimes, the information used by the RAG is private, and thus has proprietary and geographical restrictions. Thus, it is useful to design a system which does not need to send information to third parties in order to maintain data privacy. For these reasons, the tools used for this project are [Ollama](https://ollama.com) (a tool to locally deploy language models), [Chroma](https://www.trychroma.com) (a tool to locally work with vector databases), [Hugging Face](https://huggingface.co) (a set of libraries to download open-source embedding or ML models), and [LlamaIndex](https://docs.llamaindex.ai/en/stable/) (a tool to build complete LLMOps pipelines). In particular, the models recommended for the deployment of this project are mixtral (a well performing LLM with a trully open-source license) and GTE (well performing embedding model with state-of-the-art training methods).

This project is designed as follows:
- High-level configurations can be made by the user just by editing `config.yaml` according to their preferences.
- First, the documents are loaded from source (in this case, plain text files), segmented in chunks, and vectorized using a Huggingface model.
- Then the vectors are stored in a local database, which is Chroma in this case.
- Whenever the knowledge base is updated, the database can be updated as well with Chroma (run `python database_handler.py -u`).
- If any user needs to query the knowledge base, the backend loads both the vector database and a large language model.
- The query is passed through the LlamaIndex RAG pipeline and the results are returned through a RestAPI.
- The application frontend processes both the request and the response in a friendly user interface.

A simple architecture diagram looks as follows:

![Architecture diagram](architecture.svg)
