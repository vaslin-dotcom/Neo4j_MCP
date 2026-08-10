import asyncio
import sys

from mcp.client.stdio import stdio_client
from mcp import ClientSession,StdioServerParameters
from config import get_llm,uri,user_name,password
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.types import CreateMessageResult, TextContent, SamplingCapability, SamplingToolsCapability, \
    LoggingMessageNotificationParams


async def logging_callback(params: LoggingMessageNotificationParams):
    # params.level is like "info", "warning", "error", "debug"
    # params.data is the actual message payload (string, in ctx.info(msg) case)
    print(f"[server log:{params.level}] {params.data}")

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
    prompt_text = params.messages[0].content.text

    if params.tools:
        schema = params.tools[0].inputSchema
        structured_llm = get_llm(output_schema=schema)
        result = structured_llm.invoke(prompt_text)
        print(f"[sampling_callback] result type={type(result)} value={result!r}", file=sys.stderr, flush=True)
        response_text = json.dumps(result)
    else:
        structured_llm = get_llm()
        response = structured_llm.invoke(prompt_text)
        response_text = response.content

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model=structured_llm.last_model_used or "unknown",
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
                read, write,
                sampling_callback=sampling_callback,
                sampling_capabilities=SamplingCapability(tools=SamplingToolsCapability()),
                logging_callback=logging_callback,
        ) as session:
            await session.initialize()
            #await session.set_logging_level("info")
            tools = await load_mcp_tools(session)
            llm = get_llm().bind_tools(tools)

            file=r"D:\manual\1. EMD Dicer User - Manual_compressed.pdf"

            messages=[
                SystemMessage(content="You are a helpful assistant that summarizes documents concisely."),
                HumanMessage(content=f"create a graph db with entities and relationships from the file {file}"),
            ]
            summary_response=await run_agent_loop(llm,session,messages)


            print('---------Summary Response-------------')
            print(summary_response)


if __name__ == "__main__":
    asyncio.run(main())