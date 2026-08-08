import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession,StdioServerParameters
from config import get_llm,uri,user_name,password
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.types import CreateMessageResult,TextContent,SamplingCapability, SamplingToolsCapability

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

async def sampling_callback(context, params):
    prompt_text = params.messages[0].content.text

    if params.tools:
        schema = params.tools[0].inputSchema
        structured_llm = get_llm(output_schema=schema)
        result = structured_llm.invoke(prompt_text)
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
        ) as session:
            await session.initialize()

            tools=await load_mcp_tools(session)
            llm = get_llm().bind_tools(tools)

            file=r"D:\data\GOT.pdf"

            messages=[
                SystemMessage(content="You are a helpful assistant who has access to graph db."),
                HumanMessage(content="to whom is Olenna Redwyne the mother of"),
            ]
            summary_response=await run_agent_loop(llm,session,messages)


            print('---------Summary Response-------------')
            print(summary_response)


if __name__ == "__main__":
    asyncio.run(main())