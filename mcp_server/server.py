import json
from mcp.server.fastmcp import FastMCP, Context
import sys
from utils import Neo4jSetup, chunk_document_streaming
from schemas import *
from prompts import *
from mcp.types import SamplingMessage, TextContent, Tool, ToolChoice

mcp = FastMCP('Neo4j-MCP')
neo4j_instance = Neo4jSetup()

EXTRACTION_TOOL_NAME = "return_extraction"


@mcp.tool()
async def extract_and_store_entities(file_path: str, ctx: Context) -> str:
    """Extract named entities and relationships from a PDF/DOCX file and
    write them into Neo4j.

    The file is split into chunks, each chunk is processed through the
    connected client's LLM (via MCP sampling), and the combined results
    are written into the graph database once extraction is complete.
    """
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    failed_chunks = 0
    total_chunks = 0

    extraction_tool = Tool(
        name=EXTRACTION_TOOL_NAME,
        description="Return the extracted entities and relationships for this chunk of text.",
        inputSchema=ExtractionResult.model_json_schema(),
    )

    for i, chunk in enumerate(chunk_document_streaming(file_path)):
        total_chunks += 1
        prompt = EXTRACTION_PROMPT.format(chunk=chunk)
        try:
            result = await ctx.session.create_message(
                messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
                max_tokens=8000,
                tools=[extraction_tool],
                tool_choice=ToolChoice(mode="required"),
            )
            tool_call_content = result.content
            if not isinstance(tool_call_content, list):
                tool_call_content = [tool_call_content]

            for block in tool_call_content:
                if block.type == "text":
                    try:
                        data = json.loads(block.text)
                        extraction = ExtractionResult.model_validate(data)
                        all_entities.extend(extraction.entities)
                        all_relationships.extend(extraction.relationships)
                    except Exception as parse_err:
                        failed_chunks += 1
                        print(f"[chunk {i}] parse failed: {parse_err}", file=sys.stderr)
                        continue
                elif block.type == "tool_use":
                    extraction = ExtractionResult.model_validate(block.input)
                    all_entities.extend(extraction.entities)
                    all_relationships.extend(extraction.relationships)
        except Exception as e:
            failed_chunks += 1
            print(f"[chunk {i}] failed: {e}", file=sys.stderr)
            continue

    # Nothing extracted at all - don't bother calling Neo4j
    if total_chunks > 0 and failed_chunks == total_chunks:
        return json.dumps({"error": f"All {total_chunks} chunks failed extraction. Check server logs for details."})

    # Convert Pydantic objects -> plain dicts, exactly the shape write_graph expects
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
            relationship types actually stored in the graph, so exact
            wording isn't required.
    """
    result = neo4j_instance.get_entity_context(entity_query, relation_query)
    return json.dumps(result)

@mcp.tool()
def setup_neo4j_connection() -> str:
    try:
        neo4j_instance.connect()
        neo4j_instance.setup_constraints()
        return "Connected to Neo4j successfully. Ready to proceed."
    except Exception as e:
        return f"Could not connect to Neo4j: {e}"


if __name__ == '__main__':
    mcp.run(transport='stdio')