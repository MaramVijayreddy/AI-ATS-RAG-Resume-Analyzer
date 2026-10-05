import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv('LLM_PROVIDER','huggingface').lower()
    hf_token: str = os.getenv('HF_TOKEN','')
    hf_llm_model: str = os.getenv('HF_LLM_MODEL','openai/gpt-oss-120b')
    openai_api_key: str = os.getenv('OPENAI_API_KEY','')
    openai_model: str = os.getenv('OPENAI_MODEL','gpt-4.1-mini')
    embedding_model: str = os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
    chunk_size: int = int(os.getenv('CHUNK_SIZE','180'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP','40'))
    initial_top_k: int = int(os.getenv('INITIAL_TOP_K','10'))
    final_top_k: int = int(os.getenv('FINAL_TOP_K','8'))
    max_new_tokens: int = int(os.getenv('MAX_NEW_TOKENS','5000'))
    semantic_threshold: float = float(os.getenv('SEMANTIC_THRESHOLD','0.72'))

settings = Settings()
