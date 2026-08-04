import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession,StdioServerParameters
from config import get_llm,uri,user_name,password
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.types import CreateMessageResult,TextContent

async def run_agent_loop(llm,session,messages):
    for loop in range(10):
        ai_response = llm.invoke(messages)
        messages.append(ai_response)
        if not ai_response.tool_calls:
            return ai_response.content

        else:
            for call in ai_response.tool_calls:
                tool_result = await session.call_tool(call['name'], call['args'])
                tool_output = tool_result.content[0].text
                messages.append(ToolMessage(content=tool_output, tool_call_id=call['id']))

    return "stopped tool calling"


async def sampling_callback(context, params):
    """Called by the MCP SDK whenever the server sends a sampling request.
    We take the prompt the server sent, run it through our own LLM, and
    hand the result back."""
    prompt_text = params.messages[0].content.text

    llm = get_llm()  # no tools bound - this is a plain completion, not a tool-calling turn
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
        async with ClientSession(read, write,sampling_callback=sampling_callback) as session:
            await session.initialize()

            tools=await load_mcp_tools(session)
            llm = get_llm().bind_tools(tools)

            file=r"D:\data\Game+of+Thrones.pdf"

            messages=[
                SystemMessage(content="You are a helpful assistant that summarizes documents concisely."),
                HumanMessage(content=f"What are the entities and relationships in {file}"),
            ]
            summary_response=await run_agent_loop(llm,session,messages)


            print('---------Summary Response-------------')
            print(summary_response)


if __name__ == "__main__":
    asyncio.run(main())