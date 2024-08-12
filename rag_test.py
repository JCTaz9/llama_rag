"""
title: Llama Index Ollama R2R Pipeline
author: Marvin Beak
date: 2024-08-12
version: 1.0
license: MIT
description: A pipeline for retrieving and reranking relevant documents using the Llama Index library with Ollama embeddings, integrated with R2R for custom RAG.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, requests
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import requests
from pydantic import BaseModel

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        R2R_BASE_URL: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "R2R_BASE_URL": os.getenv("R2R_BASE_URL", "http://localhost:8000"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # First, retrieve relevant information from the Llama Index, then use R2R to rerank the results.

        print(messages)
        print(user_message)

        # Step 1: Retrieve relevant documents from Llama Index
        query_engine = self.index.as_query_engine(streaming=True)
        llama_response = query_engine.query(user_message)

        # Step 2: Send the Llama Index results to R2R for reranking
        response_gen = []
        for response_chunk in llama_response.response_gen:
            response_gen.append(response_chunk)
        
        r2r_response = requests.post(
            self.valves.R2R_BASE_URL + "/rerank",
            json={"query": user_message, "documents": response_gen, "top_k": 5}
        )

        if r2r_response.status_code != 200:
            return f"Error querying R2R system: {r2r_response.status_code}"

        reranked_docs = r2r_response.json().get("documents", [])

        # Step 3: Format and return the reranked results
        result = "Top 5 reranked documents:\n"
        for i, doc in enumerate(reranked_docs, 1):
            result += f"{i}. {doc['title']} - {doc['snippet']}\n"

        return result
