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

    The file is first split into chunks so the total count is known up
    front. Each chunk is then processed through the connected client's
    LLM (via MCP sampling) and its extracted entities/relationships are
    written into the graph immediately - so partial progress is never
    lost even if a later chunk fails or the run is interrupted.
    """
    import asyncio

    # Step 1: materialize chunks up front so we know the total count
    chunks = list(chunk_document_streaming(file_path))
    total_chunks = len(chunks)

    await ctx.info(f"Split '{file_path}' into {total_chunks} chunks. Starting extraction.")

    extraction_tool = Tool(
        name=EXTRACTION_TOOL_NAME,
        description="Return the extracted entities and relationships for this chunk of text.",
        inputSchema=ExtractionResult.model_json_schema(),
    )

    failed_chunks = 0
    entities_written_total = 0
    relationships_written_total = 0
    relationships_skipped_total = 0

    for i, chunk in enumerate(chunks):
        prompt = EXTRACTION_PROMPT.format(chunk=chunk)
        chunk_entities: list[Entity] = []
        chunk_relationships: list[Relationship] = []

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
                        chunk_entities.extend(extraction.entities)
                        chunk_relationships.extend(extraction.relationships)
                    except Exception as parse_err:
                        failed_chunks += 1
                        print(f"[chunk {i}] parse failed: {parse_err}", file=sys.stderr)
                        continue
                elif block.type == "tool_use":
                    extraction = ExtractionResult.model_validate(block.input)
                    chunk_entities.extend(extraction.entities)
                    chunk_relationships.extend(extraction.relationships)
        except Exception as e:
            failed_chunks += 1
            print(f"[chunk {i}] failed: {e}", file=sys.stderr)
            await ctx.info(f"Chunk {i + 1}/{total_chunks} failed extraction: {e}")
            await ctx.report_progress(progress=i + 1, total=total_chunks)
            continue

        # Step 2: write THIS chunk's entities/relationships immediately.
        # Wrapped in try/except so one bad chunk's DB write doesn't kill
        # extraction of the remaining chunks.
        if chunk_entities or chunk_relationships:
            try:
                entities_payload = [e.model_dump() for e in chunk_entities]
                relationships_payload = [r.model_dump() for r in chunk_relationships]

                # write_graph does blocking Neo4j + embedding calls - run off
                # the event loop so progress notifications keep streaming smoothly.
                write_stats = await asyncio.to_thread(
                    neo4j_instance.write_graph, entities_payload, relationships_payload
                )

                entities_written_total += write_stats["entities_written"]
                relationships_written_total += write_stats["relationships_written"]
                relationships_skipped_total += write_stats["relationships_skipped"]

                await ctx.info(
                    f"Chunk {i + 1}/{total_chunks} written: "
                    f"+{write_stats['entities_written']} entities, "
                    f"+{write_stats['relationships_written']} relationships "
                    f"(running total: {entities_written_total} entities, "
                    f"{relationships_written_total} relationships)"
                )
            except Exception as write_err:
                failed_chunks += 1
                print(f"[chunk {i}] write failed: {write_err}", file=sys.stderr)
                await ctx.info(f"Chunk {i + 1}/{total_chunks} write failed: {write_err}")
        else:
            await ctx.info(f"Chunk {i + 1}/{total_chunks}: nothing extracted.")

        await ctx.report_progress(progress=i + 1, total=total_chunks)

    if total_chunks > 0 and failed_chunks == total_chunks:
        return json.dumps({"error": f"All {total_chunks} chunks failed extraction. Check server logs for details."})

    output = {
        "total_chunks": total_chunks,
        "entities_written": entities_written_total,
        "relationships_written": relationships_written_total,
        "relationships_skipped": relationships_skipped_total,
    }
    if failed_chunks > 0:
        output["warning"] = f"{failed_chunks} of {total_chunks} chunks failed to extract - results may be incomplete."

    await ctx.info(
        f"Extraction complete: {entities_written_total} entities, "
        f"{relationships_written_total} relationships written across {total_chunks} chunks "
        f"({failed_chunks} failed)."
    )

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