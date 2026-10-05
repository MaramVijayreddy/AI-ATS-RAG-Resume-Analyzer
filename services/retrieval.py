from sklearn.metrics.pairwise import cosine_similarity

JD_QUERIES=[
'required skills and technologies','preferred skills and technologies','programming languages frameworks libraries tools','AI machine learning deep learning NLP LLM RAG','SQL databases data analytics business intelligence','cloud DevOps APIs software engineering','education experience qualifications certifications','job responsibilities duties']
RESUME_QUERIES=[
'programming languages technical skills','frameworks libraries tools platforms','AI machine learning deep learning NLP LLM RAG','SQL databases data analytics business intelligence','cloud DevOps APIs software engineering','projects technologies used','work experience internships responsibilities','education qualifications certifications']

def multi_query_retrieve(index, embedder, queries, source, initial_k=10, final_k=8):
    candidates={}
    for q in queries:
        for item in index.retrieve(q,source=source,top_k=initial_k):
            candidates[item['text']]={**item,'query':q}
    if not candidates: return []
    texts=list(candidates); qemb=embedder.encode(queries); demb=embedder.encode(texts)
    scores=cosine_similarity(demb,qemb).max(axis=1)
    results=[]
    for text,score in zip(texts,scores):
        item=candidates[text].copy(); item['rerank_score']=float(score); results.append(item)
    return sorted(results,key=lambda x:x['rerank_score'],reverse=True)[:final_k]

def build_context(results):
    return '\n\n'.join(f"--- Evidence {i} ---\nSource: {x['metadata']['source']}\nScore: {x.get('rerank_score',0):.4f}\n\n{x['text']}" for i,x in enumerate(results,1))
