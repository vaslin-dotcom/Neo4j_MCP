import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from config import uri, user_name, password
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from config import get_llm  # only used for the AGENT's own reasoning now, not extraction


async def run_agent_loop(llm, session, messages, max_tool_calls=15):
    tool_call_count = 0
    for loop in range(10):
        ai_response = llm.invoke(messages)
        messages.append(ai_response)
        if not ai_response.tool_calls:
            return ai_response.content

        for call in ai_response.tool_calls:
            if tool_call_count >= max_tool_calls:
                return "I wasn't able to find a confident answer after several attempts."
            tool_call_count += 1
            tool_result = await session.call_tool(call['name'], call['args'])
            tool_output = tool_result.content[0].text
            messages.append(ToolMessage(content=tool_output, tool_call_id=call['id']))

    return "stopped tool calling"


async def main():
    server_parameters = StdioServerParameters(
        command='python',
        args=['mcp_server/server.py'],
        env={
            "NEO4J_URI": uri,
            "NEO4J_USERNAME": user_name,
            "NEO4J_PASSWORD": password,
            # NVIDIA_API_KEY / GROQ_API_KEY now read by the SERVER itself via its own .env,
            # not passed through here - keep them in mcp_server/.env
        }
    )

    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:  # no sampling_callback, no sampling_capabilities
            await session.initialize()

            tools = await load_mcp_tools(session)
            llm = get_llm().bind_tools(tools)
            file=r"D:\data\Tamil_movies_dataset.csv"
            messages = [
                SystemMessage(content="You are a helpful assistant who has access to a graph db."),
                HumanMessage(content=f"create a db from this file {file}"),
            ]
            summary_response = await run_agent_loop(llm, session, messages)

            print('---------Summary Response-------------')
            print(summary_response)


if __name__ == "__main__":
    asyncio.run(main())