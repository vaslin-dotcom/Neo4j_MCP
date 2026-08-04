from mcp.server.fastmcp import FastMCP,Context
from utils import Neo4jSetup,chunk_document_streaming
from mcp.types import SamplingMessage,TextContent
import json
from schemas import *
from prompts import *

mcp=FastMCP('Neo4j-MCP')
neo4j_instance = Neo4jSetup()

@mcp.tool()
async def extract_entities(file_path: str, ctx: Context) -> str:
    """Extract named entities and the relationships between them from a
    PDF or DOCX file on disk.

    The file is split into chunks and each chunk is processed individually
    through the connected client's LLM (via MCP sampling), so this may take
    some time for large documents.

    Args:
        file_path: Absolute local file path to a .pdf or .docx file.

    Returns:
        A JSON string with two keys:
        - "entities": list of {name, type, description}
        - "relationships": list of {source, relation, target}
        These are ready to be written into a graph database such as Neo4j.
    """

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    schema = ExtractionResult.model_json_schema()

    for i, chunk in enumerate(chunk_document_streaming(file_path)):
        prompt = EXTRACTION_PROMPT.format(schema=schema, chunk=chunk)

        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            max_tokens=3000,
        )

        raw_text = result.content.text

        try:
            cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
            extraction = ExtractionResult.model_validate_json(cleaned)
            all_entities.extend(extraction.entities)
            all_relationships.extend(extraction.relationships)
        except Exception as e:
            print(f"[chunk {i}] failed to parse extraction: {e}")
            continue

    return json.dumps({
        "entities": [e.model_dump() for e in all_entities],
        "relationships": [r.model_dump() for r in all_relationships],
    })


@mcp.tool()
def setup_neo4j_connection() -> str:
    """Verify that a connection to Neo4j can be established using the configured credentials."""
    try:
        neo4j_instance.connect()
        return "Connected to Neo4j successfully. Ready to proceed."
    except Exception as e:
        return f"Could not connect to Neo4j: {e}"



if __name__=='__main__':
    mcp.run(transport='stdio')



