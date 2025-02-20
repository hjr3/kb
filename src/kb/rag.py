from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import os

from .obsidian import ObsidianLoader
from .config import Config, LLMType


class KnowledgeBase:
    def __init__(self, config: Config, persist_directory: str):
        self.loader = ObsidianLoader(config.obsidian.vault)
        self.persist_directory = persist_directory

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )

        if os.path.exists(persist_directory):
            print(f"Loading existing vector store from: {persist_directory}")
            self.vector_store = Chroma(
                persist_directory=persist_directory, embedding_function=self.embeddings
            )
        else:
            print("No existing vector store found. Will create new one.")
            self.vector_store = None

        temperature = 0.4
        match config.llm.llm_type:
            case LLMType.ANTHROPIC:
                self.llm = ChatAnthropic(
                    model=config.llm.model,
                    anthropic_api_key=config.llm.api_key.get_secret_value(),
                    temperature=temperature,
                )
            case LLMType.OLLAMA:
                self.llm = ChatOllama(
                    model=config.llm.model,
                    client_kwargs={
                        "headers": {"X-API-Key": config.llm.api_key.get_secret_value()}
                    },
                    base_url=config.llm.base_url,
                    temperature=temperature,
                )
            case _:
                raise ValueError(f"Unknown LLM type: {config.llm.type}")

    def initialize_vector_store(self, force_reload: bool = False) -> Optional[int]:
        """Load notes and initialize the vector store if it doesn't exist or force_reload is True"""
        if not force_reload and self.vector_store is not None:
            print("Using existing vector store")
            return None

        print("Loading documents...")
        documents = self.loader.load_notes()
        print(f"Loaded {len(documents)} documents")

        print("Splitting documents...")
        splits = self.loader.split_documents(documents)
        print(f"Created {len(splits)} chunks")

        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"Vector store persisted to: {self.persist_directory}")

        return len(splits)

    async def query(
        self, question: str, k: int = 3, filter_tags: List[str] = None
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call initialize_vector_store() first."
            )

        search_kwargs = {"k": k}
        if filter_tags:
            search_kwargs["filter"] = {"tags": {"$in": filter_tags}}

        docs = self.vector_store.similarity_search(question, **search_kwargs)

        context = "\n\n".join(
            [
                f"Title: {doc.metadata['title']}\nContent: {doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""Based on the following context from my notes, please answer this question: {question}

Context:
{context}

Please provide a clear and concise answer, citing specific notes where relevant."""

        response = await self.llm.ainvoke(prompt)

        return {
            "answer": response.content,
            "sources": [
                {
                    "title": doc.metadata["title"],
                    "source": doc.metadata["source"],
                    "tags": doc.metadata["tags"].split(),
                }
                for doc in docs
            ],
        }
