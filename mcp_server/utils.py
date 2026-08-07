import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
import re
import sys

def normalize_name(name: str) -> str:
    """Collapse whitespace and standardize casing so near-duplicate names merge into one node."""
    return re.sub(r'\s+', ' ', name.strip()).title()


def sanitize_relation_type(relation: str) -> str:
    """Convert an LLM-provided relation label into a safe Cypher relationship type."""
    cleaned = re.sub(r'[^A-Za-z0-9_]', '_', relation.strip().upper())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned or "RELATED_TO"


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

    def write_graph(self, entities: list[dict], relationships: list[dict]) -> dict:
        """
        entities: [{"name":..., "type":..., "description":...}, ...]
        relationships: [{"source": {name,type,description}, "relation":..., "target": {name,type,description}}, ...]
        """
        if self.driver is None:
            self.connect()

        norm_entities = [
            {**e, "name": normalize_name(e["name"])}
            for e in entities
        ]

        norm_relationships = []
        for r in relationships:
            norm_relationships.append({
                "source_name": normalize_name(r["source"]["name"]),
                "source_type": r["source"].get("type", "Unknown"),
                "target_name": normalize_name(r["target"]["name"]),
                "target_type": r["target"].get("type", "Unknown"),
                "relation": r["relation"],
            })

        entities_written = 0
        relationships_written = 0
        relationships_skipped = 0

        with self.driver.session() as session:
            if norm_entities:
                session.execute_write(self._merge_entities, norm_entities)
                entities_written = len(norm_entities)

            by_type: dict[str, list[dict]] = {}
            for rel in norm_relationships:
                rel_type = sanitize_relation_type(rel["relation"])
                by_type.setdefault(rel_type, []).append(rel)

            for rel_type, rels in by_type.items():
                try:
                    session.execute_write(self._merge_relationships, rel_type, rels)
                    relationships_written += len(rels)
                except Exception as e:
                    print(f"[neo4j] failed to write relation type {rel_type}: {e}", file=sys.stderr)
                    relationships_skipped += len(rels)

        return {
            "entities_written": entities_written,
            "relationships_written": relationships_written,
            "relationships_skipped": relationships_skipped,
        }

    @staticmethod
    def _merge_entities(tx, entities: list[dict]):
        tx.run(
            """
            UNWIND $entities AS e
            MERGE (n:Entity {name: e.name, type: e.type})
            SET n.description = coalesce(e.description, n.description, '')
            """,
            entities=entities,
        )

    @staticmethod
    def _merge_relationships(tx, rel_type: str, rels: list[dict]):
        # rel_type is sanitized (alnum/underscore only) before reaching here — safe to interpolate.
        # MERGE (not MATCH) on both endpoints: if a relationship references an
        # entity the extraction step never produced, create it with its stated
        # type instead of silently dropping the relationship.
        query = f"""
            UNWIND $rels AS r
            MERGE (a:Entity {{name: r.source_name, type: r.source_type}})
            MERGE (b:Entity {{name: r.target_name, type: r.target_type}})
            MERGE (a)-[rel:`{rel_type}`]->(b)
        """
        tx.run(query, rels=rels)


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