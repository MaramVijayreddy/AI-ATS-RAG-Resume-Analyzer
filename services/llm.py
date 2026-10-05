import json, re
from huggingface_hub import InferenceClient
from openai import OpenAI
from .config import settings

class LLM:
    def __init__(self):
        self.provider=settings.llm_provider
        if self.provider=='huggingface':
            if not settings.hf_token: raise ValueError('HF_TOKEN is missing. Add it to .env.')
            self.client=InferenceClient(token=settings.hf_token, provider='auto')
        elif self.provider=='openai':
            if not settings.openai_api_key: raise ValueError('OPENAI_API_KEY is missing. Add it to .env.')
            self.client=OpenAI(api_key=settings.openai_api_key)
        else: raise ValueError('LLM_PROVIDER must be huggingface or openai.')
    def generate(self, system_prompt, user_prompt, max_tokens=None):
        if self.provider=='huggingface':
            r=self.client.chat.completions.create(model=settings.hf_llm_model,messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],max_tokens=max_tokens or settings.max_new_tokens,temperature=0.1)
            return r.choices[0].message.content
        r=self.client.chat.completions.create(model=settings.openai_model,messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],max_tokens=max_tokens or settings.max_new_tokens,temperature=0.1)
        return r.choices[0].message.content

def extract_json(text):
    text=text.strip(); text=re.sub(r'^```json\s*','',text,flags=re.I); text=re.sub(r'^```\s*','',text); text=re.sub(r'\s*```$','',text)
    start,end=text.find('{'),text.rfind('}')
    if start<0 or end<0: raise ValueError('LLM did not return a JSON object.')
    return json.loads(text[start:end+1])
