import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import RateLimitError, InternalServerError, NotFoundError, APITimeoutError

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"  # verify exact slug in your NVIDIA catalog

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

uri = os.getenv("NEO4J_URI")
user_name = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")


def _build_llm(model: str, api_key: str, base_url: str, max_tokens: int = 6000, disable_thinking: bool = False):
    kwargs = dict(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        request_timeout=90,
        max_tokens=max_tokens,
    )
    if disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(**kwargs)


class SmartLLM:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.last_model_used = None

    def bind_tools(self, tools):
        return SmartLLM(
            primary=self.primary.bind_tools(tools),
            fallback=self.fallback.bind_tools(tools),
        )

    # --- sync path (kept for anywhere still calling it directly / non-async contexts) ---
    def invoke(self, prompt, retries=2):
        for attempt in range(retries + 1):
            try:
                result = self.primary.invoke(prompt)
                self.last_model_used = NVIDIA_MODEL
                return result
            except (RateLimitError, InternalServerError, NotFoundError, APITimeoutError) as e:
                print(f"[NVIDIA failed (attempt {attempt + 1}): {type(e).__name__}] {e}")
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                return self._invoke_fallback(prompt)

    def _invoke_fallback(self, prompt, retries=1):
        for attempt in range(retries + 1):
            try:
                result = self.fallback.invoke(prompt)
                self.last_model_used = GROQ_MODEL
                return result
            except RateLimitError as e:
                print(f"[Groq rate-limited (attempt {attempt + 1})] {e}")
                if attempt < retries:
                    time.sleep(12)
                    continue
                raise
            except Exception as e:
                print(f"[Groq fallback also failed] {type(e).__name__}: {e}")
                raise

    # --- async path (used for concurrent chunk processing) ---
    async def ainvoke(self, prompt, retries=2):
        for attempt in range(retries + 1):
            try:
                result = await asyncio.to_thread(self.primary.invoke, prompt)
                self.last_model_used = NVIDIA_MODEL
                return result
            except (RateLimitError, InternalServerError, NotFoundError, APITimeoutError) as e:
                print(f"[NVIDIA failed (attempt {attempt + 1}): {type(e).__name__}] {e}")
                if attempt < retries:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                return await self._ainvoke_fallback(prompt, retries=1)

    async def _ainvoke_fallback(self, prompt, retries=1):
        for attempt in range(retries + 1):
            try:
                result = await asyncio.to_thread(self.fallback.invoke, prompt)
                self.last_model_used = GROQ_MODEL
                return result
            except RateLimitError as e:
                print(f"[Groq rate-limited (attempt {attempt + 1})] {e}")
                if attempt < retries:
                    await asyncio.sleep(12)
                    continue
                raise
            except Exception as e:
                print(f"[Groq fallback also failed] {type(e).__name__}: {e}")
                raise


def get_llm(output_schema=None):
    nvidia_llm = _build_llm(NVIDIA_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, disable_thinking=True)
    groq_llm = _build_llm(GROQ_MODEL, GROQ_API_KEY, GROQ_BASE_URL, max_tokens=2000)

    if output_schema:
        primary_final = nvidia_llm.with_structured_output(output_schema)
        fallback_final = groq_llm.with_structured_output(output_schema, method="function_calling")
    else:
        primary_final = nvidia_llm
        fallback_final = groq_llm

    return SmartLLM(primary=primary_final, fallback=fallback_final)


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Hello, who are you")
    print(response.content)