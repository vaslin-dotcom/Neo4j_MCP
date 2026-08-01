#LLM
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
def groq_llm(tools=None):
    key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1"
    llm = ChatOpenAI(model='llama-3.3-70b-versatile', temperature=0.1,api_key=key,base_url=url)
    if tools:
        return llm.bind_tools(tools)   # Bind tools so LLM can emit tool_call blocks
    return llm

if __name__ == "__main__":
    llm=groq_llm()
    print(llm.invoke("Hello").content)