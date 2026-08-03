import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class Neo4jSetup:
    def __init__(self, uri=None, username=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self):
        """Create the driver and verify the connection actually works."""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            self.driver.verify_connectivity()
            print("Neo4j connection successful.")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            raise

    def setup_constraints(self):
        """
        Ensure Entity (name, type) combinations are unique - this is what makes
        MERGE work correctly for exact-match deduplication (v1 approach,
        no embedding similarity yet). Using the composite key instead of
        name alone so "Apple" (Organization) and "Apple" (Concept) are
        correctly treated as distinct nodes.
        """
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
            """)
        print("Entity (name, type) uniqueness constraint ensured.")


from typing import Iterator
from pypdf import PdfReader
from docx import Document
from pathlib import Path



def stream_pages(file_path: str) -> Iterator[str]:
    """Yield text page-by-page (PDF) or paragraph-block-by-block (DOCX)
    instead of loading the whole document into one string."""
    path = Path(file_path)
    if path.suffix == ".pdf":
        reader = PdfReader(str(path))
        for page in reader.pages:
            yield page.extract_text() or ""
    elif path.suffix == ".docx":
        document = Document(str(path))
        # batch paragraphs to approximate "page-sized" units
        buf = []
        for p in document.paragraphs:
            buf.append(p.text)
            if len("\n".join(buf)) > 2000:
                yield "\n".join(buf)
                buf = []
        if buf:
            yield "\n".join(buf)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def chunk_document_streaming(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Iterator[str]:
    """Stream chunks from a large document without holding the full text
    or full chunk list in memory at once. Suitable for 1000s of pages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    buffer = ""
    # keep buffer roughly 3x chunk_size before splitting, so the splitter
    # has enough context to find good paragraph/sentence boundaries
    flush_threshold = chunk_size * 3

    for page_text in stream_pages(file_path):
        buffer += ("\n\n" + page_text if buffer else page_text)

        if len(buffer) >= flush_threshold:
            pieces = splitter.split_text(buffer)
            # hold back the last piece — it may be incomplete without
            # the next page's text, so carry it into the next round
            for piece in pieces[:-1]:
                yield piece
            buffer = pieces[-1] if pieces else ""

    # flush whatever's left after the last page
    if buffer:
        for piece in splitter.split_text(buffer):
            yield piece