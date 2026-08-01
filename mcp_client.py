import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession,StdioServerParameters
from config import groq_llm
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools

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


async def main():
    server_parameters = StdioServerParameters(
        command='python',
        args=['mcp_server.py']
    )

    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools=await load_mcp_tools(session)
            llm=groq_llm(tools=tools)

            file=r"D:\manual\1. EMD Dicer User - Manual_compressed.pdf"

            messages=[
                SystemMessage(content="You are a helpful assistant that summarizes documents concisely."),
                HumanMessage(content=f"summarise the following document\n{file}"),
            ]
            summary_response=await run_agent_loop(llm,session,messages)


            print('---------Summary Response-------------')
            print(summary_response)


if __name__ == "__main__":
    asyncio.run(main())