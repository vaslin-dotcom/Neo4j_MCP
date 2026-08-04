import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CreateMessageResult, TextContent
from config import groq_llm, uri, user_name, password


async def sampling_callback(context, params):
    prompt_text = params.messages[0].content.text
    llm = groq_llm()
    response = llm.invoke(prompt_text)
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response.content),
        model="llama-3.3-70b-versatile",
    )


async def main():
    server_parameters = StdioServerParameters(
        command='python',
        args=['mcp_server/server.py'],
        env={
            "NEO4J_URI": uri,
            "NEO4J_USERNAME": user_name,
            "NEO4J_PASSWORD": password,
        }
    )

    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            await session.initialize()

            file_path = r"D:\manual\1. EMD Dicer User - Manual_compressed.pdf"

            result = await session.call_tool("extract_entities", {"file_path": file_path})

            print("--- Raw Result ---")
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())