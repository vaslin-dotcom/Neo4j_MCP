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
async def extract_and_store_entities(
    file_path: str,
    ctx: Context,
    max_concurrency: int = 15,
) -> str:
    """Extract named entities and relationships from a PDF/DOCX file and
    write them into Neo4j.

    The file is first split into chunks so the total count is known up
    front. Chunks are then processed concurrently (up to `max_concurrency`
    at a time) - each chunk goes through LLM sampling and its results are
    written to the graph as soon as that chunk finishes, so partial
    progress is never lost and chunks don't block each other waiting on
    the LLM or the DB.
    """
    import asyncio

    chunks = list(chunk_document_streaming(file_path))
    total_chunks = len(chunks)

    await ctx.info(
        f"Split '{file_path}' into {total_chunks} chunks. "
        f"Starting extraction with up to {max_concurrency} chunks in parallel."
    )

    extraction_tool = Tool(
        name=EXTRACTION_TOOL_NAME,
        description="Return the extracted entities and relationships for this chunk of text.",
        inputSchema=ExtractionResult.model_json_schema(),
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    stats_lock = asyncio.Lock()

    # Dedicated counter: how many chunks have finished (success or fail),
    # regardless of order. This is what drives progress notifications.
    chunks_completed = 0

    failed_chunks = 0
    entities_written_total = 0
    relationships_written_total = 0
    relationships_skipped_total = 0

    async def process_chunk(i: int, chunk: str) -> None:
        nonlocal chunks_completed, failed_chunks
        nonlocal entities_written_total, relationships_written_total, relationships_skipped_total

        async with semaphore:
            prompt = EXTRACTION_PROMPT.format(chunk=chunk)
            chunk_entities: list[Entity] = []
            chunk_relationships: list[Relationship] = []
            chunk_failed = False

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
                            chunk_failed = True
                            print(f"[chunk {i}] parse failed: {parse_err}", file=sys.stderr)
                    elif block.type == "tool_use":
                        extraction = ExtractionResult.model_validate(block.input)
                        chunk_entities.extend(extraction.entities)
                        chunk_relationships.extend(extraction.relationships)
            except Exception as e:
                chunk_failed = True
                print(f"[chunk {i}] extraction failed: {e}", file=sys.stderr)

            write_stats = {"entities_written": 0, "relationships_written": 0, "relationships_skipped": 0}
            if not chunk_failed and (chunk_entities or chunk_relationships):
                try:
                    entities_payload = [e.model_dump() for e in chunk_entities]
                    relationships_payload = [r.model_dump() for r in chunk_relationships]
                    write_stats = await asyncio.to_thread(
                        neo4j_instance.write_graph, entities_payload, relationships_payload
                    )
                except Exception as write_err:
                    chunk_failed = True
                    print(f"[chunk {i}] write failed: {write_err}", file=sys.stderr)

            # Increment the shared counter and emit progress under the lock,
            # so concurrent chunks never interleave their updates or logs.
            async with stats_lock:
                chunks_completed += 1
                if chunk_failed:
                    failed_chunks += 1
                entities_written_total += write_stats["entities_written"]
                relationships_written_total += write_stats["relationships_written"]
                relationships_skipped_total += write_stats["relationships_skipped"]

                if chunk_failed:
                    await ctx.info(f"[{chunks_completed}/{total_chunks} done] Chunk {i + 1} failed.")
                elif write_stats["entities_written"] or write_stats["relationships_written"]:
                    await ctx.info(
                        f"[{chunks_completed}/{total_chunks} done] Chunk {i + 1} written: "
                        f"+{write_stats['entities_written']} entities, "
                        f"+{write_stats['relationships_written']} relationships "
                        f"(running total: {entities_written_total} entities, "
                        f"{relationships_written_total} relationships)"
                    )
                else:
                    await ctx.info(f"[{chunks_completed}/{total_chunks} done] Chunk {i + 1}: nothing extracted.")

                await ctx.report_progress(progress=chunks_completed, total=total_chunks)

    await asyncio.gather(*(process_chunk(i, chunk) for i, chunk in enumerate(chunks)))

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