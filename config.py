#LLM
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
import time
from openai import RateLimitError, InternalServerError

# LLM
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import RateLimitError, InternalServerError, NotFoundError

load_dotenv()

# nvidia
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

uri = os.getenv("NEO4J_URI")
user_name = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")


def _build_llm(model: str, api_key: str, base_url: str):
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        request_timeout=60
    )


class SmartLLM:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def bind_tools(self, tools):
        return SmartLLM(
            primary=self.primary.bind_tools(tools),
            fallback=self.fallback.bind_tools(tools),
        )

    def invoke(self, prompt):
        try:
            return self.primary.invoke(prompt)
        except (RateLimitError, InternalServerError, NotFoundError) as e:
            print(f"[NVIDIA failed: {type(e).__name__}] switching to Groq fallback")
            time.sleep(3)
            return self._invoke_fallback(prompt)

    def _invoke_fallback(self, prompt):
        try:
            return self.fallback.invoke(prompt)
        except Exception as e:
            print(f"[Groq fallback also failed] {type(e).__name__}: {e}")
            raise


def get_llm(output_schema=None):
    time.sleep(1.5)

    nvidia_llm = _build_llm(NVIDIA_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL)
    groq_llm = _build_llm(GROQ_MODEL, GROQ_API_KEY, GROQ_BASE_URL)

    if output_schema:
        primary_final = nvidia_llm.with_structured_output(output_schema)
        fallback_final = groq_llm.with_structured_output(output_schema, method="function_calling")
    else:
        primary_final = nvidia_llm
        fallback_final = groq_llm

    return SmartLLM(primary=primary_final, fallback=fallback_final)


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Hello")
    print(response.content)
