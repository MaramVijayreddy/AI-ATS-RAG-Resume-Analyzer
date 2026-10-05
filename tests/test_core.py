from services.pdf_utils import clean_text,chunk_text
from services.llm import extract_json

def test_clean_text(): assert clean_text('a\n  b\x00 c')=='a b c'
def test_chunk_overlap():
    x=chunk_text(' '.join(str(i) for i in range(500)),180,40)
    assert len(x)>=3 and x[0].split()[-40:]==x[1].split()[:40]
def test_json_extract(): assert extract_json('```json\n{"a": 1}\n```')['a']==1
