from .config import settings
from .retrieval import JD_QUERIES,RESUME_QUERIES,multi_query_retrieve,build_context
from .prompts import *
from .llm import extract_json
from sklearn.metrics.pairwise import cosine_similarity

def flatten_string_lists(data):
    values=[]
    for items in data.values():
        if isinstance(items,list):
            for item in items:
                if isinstance(item,str) and item.strip(): values.append(item.strip())
    return sorted(set(values),key=str.lower)

def semantic_match(jd_terms,resume_terms,embedder,threshold=None):
    threshold=threshold if threshold is not None else settings.semantic_threshold
    if not jd_terms: return {'exact_matches':[],'semantic_matches':[],'missing_terms':[],'details':[]}
    if not resume_terms: return {'exact_matches':[],'semantic_matches':[],'missing_terms':jd_terms,'details':[]}
    jd_emb=embedder.encode(jd_terms); res_emb=embedder.encode(resume_terms)
    import numpy as np
    sim=cosine_similarity(jd_emb,res_emb)
    exact=[]; semantic=[]; missing=[]; details=[]
    lookup={x.lower():x for x in resume_terms}
    for i,term in enumerate(jd_terms):
        if term.lower() in lookup:
            exact.append({'jd_term':term,'resume_term':lookup[term.lower()]}); details.append({'jd_term':term,'resume_term':lookup[term.lower()],'match_type':'exact','score':1.0})
        else:
            j=int(np.argmax(sim[i])); score=float(sim[i,j]); r=resume_terms[j]
            if score>=threshold:
                semantic.append({'jd_term':term,'resume_term':r,'score':score}); details.append({'jd_term':term,'resume_term':r,'match_type':'semantic','score':score})
            else:
                missing.append(term); details.append({'jd_term':term,'resume_term':r,'match_type':'missing','score':score})
    return {'exact_matches':exact,'semantic_matches':semantic,'missing_terms':missing,'details':details}

def run_analysis(jd_text,resume_text,index,embedder,llm):
    jd_results=multi_query_retrieve(index,embedder,JD_QUERIES,'job_description',settings.initial_top_k,settings.final_top_k)
    resume_results=multi_query_retrieve(index,embedder,RESUME_QUERIES,'resume',settings.initial_top_k,settings.final_top_k)
    jd_context=build_context(jd_results); resume_context=build_context(resume_results)
    jd_profile=extract_json(llm.generate(JD_SYSTEM,jd_prompt(jd_text,jd_context)))
    resume_profile=extract_json(llm.generate(RESUME_SYSTEM,resume_prompt(resume_context)))
    jd_keywords=flatten_string_lists(jd_profile); resume_terms=flatten_string_lists(resume_profile)
    matching=semantic_match(jd_keywords,resume_terms,embedder)
    ats=extract_json(llm.generate(ATS_SYSTEM,ats_prompt(jd_profile,resume_profile,matching,jd_context,resume_context)))
    total=len(jd_keywords); exact=len(matching['exact_matches']); semantic=len(matching['semantic_matches'])
    coverage={'jd_terms':total,'exact_percent':round(exact/total*100,2) if total else 0,'exact_plus_semantic_percent':round((exact+semantic)/total*100,2) if total else 0}
    return {'jd_profile':jd_profile,'resume_profile':resume_profile,'semantic_matching':matching,'coverage':coverage,'ats_analysis':ats,'retrieved':{'jd':jd_results,'resume':resume_results}}
