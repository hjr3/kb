from pathlib import Path
import re
from typing import List
import yaml
from datetime import datetime
from langchain.schema import Document
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ObsidianLoader:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)

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

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from markdown content"""
        frontmatter = {}
        markdown_content = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    markdown_content = parts[2]
                except yaml.YAMLError:
                    pass

        return frontmatter, markdown_content

    def _extract_tags(self, content: str) -> List[str]:
        """Extract Obsidian tags from content"""
        tag_pattern = r"#[\w/-]+"
        return list(set(re.findall(tag_pattern, content)))

    def _process_internal_links(self, content: str) -> str:
        """Convert Obsidian internal links to plain text"""
        content = re.sub(r"\[\[(.*?)\]\]", r"\1", content)
        content = re.sub(r"\[\[(.*?)\|(.*?)\]\]", r"\2", content)
        return content

    def load_notes(self) -> List[Document]:
        """Load all markdown files from the vault"""
        documents = []

        md_files = list(self.vault_path.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")

        for file_path in md_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                frontmatter, markdown_content = self._parse_frontmatter(content)
                processed_content = self._process_internal_links(markdown_content)

                tags = self._extract_tags(content)
                if "tags" in frontmatter:
                    if isinstance(frontmatter["tags"], list):
                        tags.extend(frontmatter["tags"])
                    else:
                        tags.append(frontmatter["tags"])

                # Convert tags list to string for Chroma compatibility
                tags_str = " ".join(tags)

                metadata = {
                    "source": str(file_path.relative_to(self.vault_path)),
                    "title": file_path.stem,
                    "tags": tags_str,
                    "created": frontmatter.get("created", ""),
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                }

                # Add any scalar frontmatter values
                for key, value in frontmatter.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value

                doc = Document(page_content=processed_content, metadata=metadata)
                documents.append(doc)
                print(f"Processed: {file_path.name}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split large documents into smaller chunks"""

        splits = []
        for doc in documents:
            doc_splits = self._split_document(doc)
            splits.extend(doc_splits)
        return splits

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
