import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import RateLimitError, InternalServerError, NotFoundError, APITimeoutError

load_dotenv()

# nvidia
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"   # verify exact slug on build.nvidia.com
NVIDIA_EMBEDDING_MODEL="nvidia/nv-embed-v1"

# groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

uri = os.getenv("NEO4J_URI")
user_name = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")


def _build_llm(model: str, api_key: str, base_url: str, max_tokens: int = 8000, disable_thinking: bool = False):
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
        kwargs["model_kwargs"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    return ChatOpenAI(**kwargs)


class SmartLLM:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.last_model_used = None   # so sampling_callback can report the true source

    def bind_tools(self, tools):
        return SmartLLM(
            primary=self.primary.bind_tools(tools),
            fallback=self.fallback.bind_tools(tools),
        )

    def invoke(self, prompt):
        start = time.time()
        try:
            result = self.primary.invoke(prompt)
            print(f"[NVIDIA succeeded in {time.time()-start:.2f}s]")
            self.last_model_used = NVIDIA_MODEL
            return result
        except (RateLimitError, InternalServerError, NotFoundError, APITimeoutError) as e:
            elapsed = time.time() - start
            print(f"[NVIDIA failed after {elapsed:.2f}s: {type(e).__name__}] {e}")
            time.sleep(3)
            return self._invoke_fallback(prompt)

    def _invoke_fallback(self, prompt):
        start = time.time()
        try:
            result = self.fallback.invoke(prompt)
            print(f"[Groq succeeded in {time.time()-start:.2f}s]")
            self.last_model_used = GROQ_MODEL
            return result
        except Exception as e:
            print(f"[Groq fallback also failed] {type(e).__name__}: {e}")
            raise


def get_llm(output_schema=None):
    time.sleep(1.5)

    nvidia_llm = _build_llm(NVIDIA_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, disable_thinking=True)
    groq_llm = _build_llm(GROQ_MODEL, GROQ_API_KEY, GROQ_BASE_URL)

    if output_schema:
        primary_final = nvidia_llm.with_structured_output(output_schema)
        fallback_final = groq_llm.with_structured_output(output_schema, method="function_calling")
    else:
        primary_final = nvidia_llm
        fallback_final = groq_llm

    return SmartLLM(primary=primary_final, fallback=fallback_final)