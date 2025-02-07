from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain.schema import Document
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

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
                ("####", "header4"),
                ("#####", "header5"),
            ],
            strip_headers=False,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
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

    def _split_document(self, doc: Document) -> List[Document]:
        """Split a document using header-based splitting first, then chunk if needed"""
        try:
            splits = self.header_splitter.split_text(doc.page_content)

            documents = []
            for split in splits:
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update(
                    {
                        f"header_{level}": split.metadata.get(level, "")
                        for level in [
                            "header1",
                            "header2",
                            "header3",
                            "header4",
                            "header5",
                        ]
                    }
                )

                # If the chunk is still too large, split it further
                if len(split.page_content) > 500:
                    subsplits = self.text_splitter.split_text(split.page_content)
                    for subsplit in subsplits:
                        documents.append(
                            Document(page_content=subsplit, metadata=enhanced_metadata)
                        )
                else:
                    documents.append(
                        Document(
                            page_content=split.page_content, metadata=enhanced_metadata
                        )
                    )

            return documents

        except Exception as e:
            print(f"Error splitting document {doc.metadata.get('source')}: {e}")
            return self.text_splitter.split_documents([doc])

    def initialize_vector_store(self, force_reload: bool = False) -> Optional[int]:
        """Load notes and initialize the vector store if it doesn't exist or force_reload is True"""
        if not force_reload and self.vector_store is not None:
            print("Using existing vector store")
            return None

        print("Loading documents...")
        documents = self.loader.load_notes()
        print(f"Loaded {len(documents)} documents")

        print("Splitting documents...")
        splits = []
        for doc in documents:
            doc_splits = self._split_document(doc)
            splits.extend(doc_splits)
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
