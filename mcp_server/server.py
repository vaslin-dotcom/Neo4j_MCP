import asyncio
import json
import sys
from mcp.server.fastmcp import FastMCP
from utils import Neo4jSetup, chunk_document_streaming
from schemas import *
from prompts import *
from config import get_llm

mcp = FastMCP('Neo4j-MCP')
neo4j_instance = Neo4jSetup()


@mcp.tool()
async def extract_and_store_entities(file_path: str) -> str:
    """Extract named entities and relationships from a PDF/DOCX/XLSX/CSV file
    and write them into Neo4j. Processes chunks concurrently for speed.
    """
    chunks = list(chunk_document_streaming(file_path))
    total_chunks = len(chunks)

    if total_chunks == 0:
        return json.dumps({"error": "No content extracted from file - it may be empty or unreadable."})

    semaphore = asyncio.Semaphore(5)  # tune based on NVIDIA/Groq capacity

    async def process_chunk(i: int, chunk: str):
        async with semaphore:
            prompt = EXTRACTION_PROMPT.format(chunk=chunk)
            structured_llm = get_llm(output_schema=ExtractionResult)
            try:
                result = await structured_llm.ainvoke(prompt)
                return result.entities, result.relationships, None
            except Exception as e:
                print(f"[chunk {i}] failed: {e}", file=sys.stderr)
                return [], [], e

    results = await asyncio.gather(*(process_chunk(i, c) for i, c in enumerate(chunks)))

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    failed_chunks = 0

    for entities, relationships, error in results:
        if error:
            failed_chunks += 1
        else:
            all_entities.extend(entities)
            all_relationships.extend(relationships)

    if failed_chunks == total_chunks:
        return json.dumps({"error": f"All {total_chunks} chunks failed extraction. Check server logs for details."})

    entities_payload = [e.model_dump() for e in all_entities]
    relationships_payload = [r.model_dump() for r in all_relationships]

    write_stats = neo4j_instance.write_graph(entities_payload, relationships_payload)

    output = {
        "entities_extracted": len(entities_payload),
        "relationships_extracted": len(relationships_payload),
        **write_stats,
    }
    if failed_chunks > 0:
        output["warning"] = f"{failed_chunks} of {total_chunks} chunks failed to extract - results may be incomplete."

    return json.dumps(output)


@mcp.tool()
def get_entity_context(entity_query: str, relation_query: str = None) -> str:
    """Look up an entity in the graph and return its connections.

    Uses semantic (embedding-based) matching, so exact spelling isn't
    required - e.g. "curie" will match "Marie Curie".

    Args:
        entity_query: The entity to look up, as mentioned in the user's question.
        relation_query: Optional - a relationship word/phrase from the question
            (e.g. "married", "works with"). Matched semantically against
            relationship types actually stored in the graph.
    """
    result = neo4j_instance.get_entity_context(entity_query, relation_query)
    return json.dumps(result)


@mcp.tool()
def setup_neo4j_connection() -> str:
    """Verify the Neo4j connection and ensure constraints/indexes exist."""
    try:
        neo4j_instance.connect()
        neo4j_instance.setup_constraints()
        return "Connected to Neo4j successfully. Ready to proceed."
    except Exception as e:
        return f"Could not connect to Neo4j: {e}"


if __name__ == '__main__':
    mcp.run(transport='stdio')