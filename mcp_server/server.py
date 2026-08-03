from mcp.server.fastmcp import FastMCP
from utils import Neo4jSetup,chunk_document_streaming

mcp=FastMCP('Neo4j-MCP')




neo4j_instance = Neo4jSetup()

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



