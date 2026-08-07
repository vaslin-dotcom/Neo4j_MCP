import os, time, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_server.schemas import ExtractionResult
from mcp_server.prompts import EXTRACTION_PROMPT

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"  # your current NVIDIA model

llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    max_retries=0,
    request_timeout=90,
    max_tokens=6000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

structured_llm = llm.with_structured_output(ExtractionResult)

sample_chunk = (
    "Saad Khan serves as Committee Director. He is a student at the "
    "University of Toronto majoring in Human Biology and Chemistry."
) * 5

prompt = EXTRACTION_PROMPT.format(chunk=sample_chunk)

start = time.time()
try:
    result = structured_llm.invoke(prompt)
    print(f"SUCCESS in {time.time()-start:.2f}s\n")

    # pretty-print as JSON so nested structure is easy to read
    print(json.dumps(result.model_dump(), indent=2))

except Exception as e:
    print(f"FAILED after {time.time()-start:.2f}s")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback; traceback.print_exc()


