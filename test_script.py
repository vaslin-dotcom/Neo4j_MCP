import os, time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_server.schemas import ExtractionResult  # your real schema

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    max_retries=0,
    request_timeout=90,   # generous, matching your working test
    max_tokens=3000,      # matching your real extraction call
)

structured_llm = llm.with_structured_output(ExtractionResult)

sample_chunk = "Saad Khan serves as Committee Director. He is a student at the University of Toronto majoring in Human Biology and Chemistry." * 5  # something chunk-sized

prompt = f"Extract entities and relationships from this text:\n{sample_chunk}"

start = time.time()
try:
    result = structured_llm.invoke(prompt)
    print(f"SUCCESS in {time.time()-start:.2f}s")
    print(result)
except Exception as e:
    print(f"FAILED after {time.time()-start:.2f}s")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"Cause: {e.__cause__}")
    import traceback; traceback.print_exc()