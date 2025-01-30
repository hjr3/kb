import os
from pathlib import Path
import re
from typing import List, Dict, Any, Optional
import yaml
from datetime import datetime

from langchain.text_splitter import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain.schema import Document

class ObsidianLoader:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        
    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from markdown content"""
        frontmatter = {}
        markdown_content = content
        
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    markdown_content = parts[2]
                except yaml.YAMLError:
                    pass
                    
        return frontmatter, markdown_content

    def _extract_tags(self, content: str) -> List[str]:
        """Extract Obsidian tags from content"""
        tag_pattern = r'#[\w/-]+'
        return list(set(re.findall(tag_pattern, content)))

    def _process_internal_links(self, content: str) -> str:
        """Convert Obsidian internal links to plain text"""
        content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)
        content = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\2', content)
        return content

    def load_notes(self) -> List[Document]:
        """Load all markdown files from the vault"""
        documents = []
        
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")
        
        for file_path in md_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                frontmatter, markdown_content = self._parse_frontmatter(content)
                processed_content = self._process_internal_links(markdown_content)
                
                tags = self._extract_tags(content)
                if 'tags' in frontmatter:
                    if isinstance(frontmatter['tags'], list):
                        tags.extend(frontmatter['tags'])
                    else:
                        tags.append(frontmatter['tags'])
                
                # Convert tags list to string for Chroma compatibility
                tags_str = ' '.join(tags)
                
                metadata = {
                    'source': str(file_path.relative_to(self.vault_path)),
                    'title': file_path.stem,
                    'tags': tags_str,  # Store as space-separated string
                    'created': frontmatter.get('created', ''),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # Add any scalar frontmatter values
                for key, value in frontmatter.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                
                doc = Document(
                    page_content=processed_content,
                    metadata=metadata
                )
                documents.append(doc)
                print(f"Processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        return documents

class ObsidianRAG:
    def __init__(
        self, 
        vault_path: str, 
        anthropic_api_key: str,
        persist_directory: str
    ):
        self.loader = ObsidianLoader(vault_path)
        self.persist_directory = persist_directory
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        if os.path.exists(persist_directory):
            print(f"Loading existing vector store from: {persist_directory}")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("No existing vector store found. Will create new one.")
            self.vector_store = None
        
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=anthropic_api_key,
            temperature=0
        )

    def initialize_vector_store(self, force_reload: bool = False) -> Optional[int]:
        """Load notes and initialize the vector store if it doesn't exist or force_reload is True"""
        if not force_reload and self.vector_store is not None:
            print("Using existing vector store")
            return None
            
        print("Loading documents...")
        documents = self.loader.load_notes()
        print(f"Loaded {len(documents)} documents")
        
        print("Splitting documents into chunks...")
        splits = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            splits.extend([
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ) for chunk in chunks
            ])
        print(f"Created {len(splits)} chunks")
        
        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Vector store persisted to: {self.persist_directory}")
        
        return len(splits)

    async def query(
        self,
        question: str,
        k: int = 3,
        filter_tags: List[str] = None
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        search_kwargs = {"k": k}
        if filter_tags:
            tags_str = ' '.join(filter_tags)
            search_kwargs["filter"] = {"tags": {"$contains": tags_str}}
        
        docs = self.vector_store.similarity_search(
            question,
            **search_kwargs
        )
        
        context = "\n\n".join([
            f"Title: {doc.metadata['title']}\nContent: {doc.page_content}"
            for doc in docs
        ])

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
                    "tags": doc.metadata["tags"].split()
                }
                for doc in docs
            ]
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    import asyncio
    
    load_dotenv()

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise ValueError("Please set the OBSIDIAN_VAULT_PATH environment variable")

    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Initializing RAG system...")
    rag = ObsidianRAG(
        vault_path=vault_path,
        anthropic_api_key=anthropic_api_key,
        persist_directory="./chroma_db"
    )
    
    rag.initialize_vector_store(force_reload=False)
    
    async def interactive_loop():
        print("\nEnter your questions (type 'quit' to exit):")
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                response = await rag.query(question)
                print("\nAnswer:", response["answer"])
                print("\nSources:")
                for source in response["sources"]:
                    print(f"- {source['title']}")
                    print(f"  Tags: {' '.join(source['tags'])}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
    
    asyncio.run(interactive_loop())
