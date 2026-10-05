import uuid
import chromadb

class RAGIndex:
    def __init__(self, embedder):
        self.embedder=embedder
        self.client=chromadb.Client()
        self.collection=self.client.get_or_create_collection('ats_resume_jd')

    def reset(self):
        try: self.client.delete_collection('ats_resume_jd')
        except Exception: pass
        self.collection=self.client.get_or_create_collection('ats_resume_jd')

    def add_documents(self, resume_chunks, jd_chunks):
        docs=[]; metas=[]; ids=[]
        for i,c in enumerate(resume_chunks):
            docs.append(c); metas.append({'source':'resume','chunk_id':i}); ids.append(f'resume_{i}_{uuid.uuid4().hex[:8]}')
        for i,c in enumerate(jd_chunks):
            docs.append(c); metas.append({'source':'job_description','chunk_id':i}); ids.append(f'jd_{i}_{uuid.uuid4().hex[:8]}')
        if not docs: return 0
        embs=self.embedder.encode(docs).tolist()
        self.collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        return len(docs)

    def retrieve(self, query, source=None, top_k=10):
        q=self.embedder.encode([query])[0].tolist()
        kwargs={'query_embeddings':[q], 'n_results':min(top_k,max(1,self.collection.count()))}
        if source: kwargs['where']={'source':source}
        result=self.collection.query(**kwargs)
        out=[]
        for i,text in enumerate(result.get('documents',[[]])[0]):
            out.append({'text':text,'metadata':result['metadatas'][0][i], 'distance':result['distances'][0][i] if result.get('distances') else None})
        return out
