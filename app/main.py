import json, tempfile, shutil
from pathlib import Path
import streamlit as st
from services.config import settings
from services.pdf_utils import extract_pdf_text,clean_text,chunk_text
from services.embeddings import Embedder
from services.vector_store import RAGIndex
from services.llm import LLM
from services.analyzer import run_analysis
from services.latex import generate_latex,compile_latex

st.set_page_config(page_title='AI ATS Resume Analyzer',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--bg:#070a10;--card:#101620;--muted:#8f9aaa;--line:#202a38;--accent:#7c8cff}
.stApp{background:linear-gradient(135deg,#070a10 0%,#0b1019 60%,#0b0d15 100%);color:#eef2f7}
.block-container{max-width:1250px;padding-top:2rem}
.hero{padding:32px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,rgba(124,140,255,.16),rgba(16,22,32,.92));margin-bottom:22px}
.hero h1{font-size:42px;margin:0}.hero p{color:#aeb7c5;font-size:17px;max-width:800px}
.card{background:rgba(16,22,32,.82);border:1px solid var(--line);border-radius:18px;padding:20px;margin:8px 0}
.metric{font-size:30px;font-weight:700}.muted{color:var(--muted)}
.badge{display:inline-block;padding:5px 9px;border-radius:999px;background:#192234;border:1px solid #2b3850;margin:3px;font-size:12px}
</style>''',unsafe_allow_html=True)

if 'analysis' not in st.session_state: st.session_state.analysis=None
if 'raw_resume' not in st.session_state: st.session_state.raw_resume=''

st.markdown('<div class="hero"><h1>AI ATS Resume Analyzer</h1><p>RAG-powered resume analysis that extracts requirements from the actual JD, retrieves evidence from your resume, performs semantic matching, and safely tailors a LaTeX resume.</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header('Configuration')
    st.caption(f'LLM provider: {settings.llm_provider}')
    st.caption(f'Embedding: {settings.embedding_model}')
    st.caption('Embeddings and Chroma run locally. API calls are used only for generation.')
    if st.button('Clear session',use_container_width=True):
        st.session_state.clear(); st.rerun()

left,right=st.columns([1,1])
with left:
    st.subheader('1. Resume PDF')
    upload=st.file_uploader('Drag and drop your resume',type=['pdf'])
    if upload:
        st.success(f'Loaded: {upload.name}')
with right:
    st.subheader('2. Job Description')
    jd=st.text_area('Paste the complete JD',height=230,placeholder='Paste the full job description here...')

if st.button('Run RAG ATS Analysis',type='primary',use_container_width=True,disabled=not(upload and jd.strip())):
    progress=st.progress(0,text='Preparing input...')
    try:
        tmp=Path(tempfile.mkdtemp(prefix='ats_')); pdf=tmp/upload.name; pdf.write_bytes(upload.getvalue())
        raw=extract_pdf_text(pdf); clean=clean_text(raw); st.session_state.raw_resume=clean
        progress.progress(15,text='Extracting resume and JD...')
        rchunks=chunk_text(clean,settings.chunk_size,settings.chunk_overlap); jchunks=chunk_text(clean_text(jd),settings.chunk_size,settings.chunk_overlap)
        progress.progress(30,text='Loading local embedding model...')
        embedder=Embedder(settings.embedding_model); index=RAGIndex(embedder); index.reset(); index.add_documents(rchunks,jchunks)
        progress.progress(50,text='Retrieving and reranking evidence...')
        llm=LLM(); progress.progress(65,text='Extracting dynamic JD and resume profiles...')
        result=run_analysis(clean_text(jd),clean,index,embedder,llm)
        progress.progress(90,text='Generating structured ATS analysis...'); st.session_state.analysis=result; st.session_state.index=index; st.session_state.embedder=embedder; st.session_state.llm=llm
        progress.progress(100,text='Analysis complete')
    except Exception as e:
        st.error(f'Analysis failed: {e}')

result=st.session_state.analysis
if result:
    c=result['coverage']; a=result['ats_analysis']
    st.divider(); st.subheader('ATS Analysis Dashboard')
    m1,m2,m3,m4=st.columns(4)
    m1.markdown(f'<div class="card"><div class="muted">JD Coverage</div><div class="metric">{c["exact_percent"]}%</div></div>',unsafe_allow_html=True)
    m2.markdown(f'<div class="card"><div class="muted">Semantic Coverage</div><div class="metric">{c["exact_plus_semantic_percent"]}%</div></div>',unsafe_allow_html=True)
    m3.markdown(f'<div class="card"><div class="muted">Exact Matches</div><div class="metric">{len(result["semantic_matching"]["exact_matches"])}</div></div>',unsafe_allow_html=True)
    m4.markdown(f'<div class="card"><div class="muted">Missing Terms</div><div class="metric">{len(result["semantic_matching"]["missing_terms"])}</div></div>',unsafe_allow_html=True)
    st.info('Coverage is a project-specific JD term coverage metric, not an employer’s official ATS score.')
    tabs=st.tabs(['Summary','Skills','Projects & Experience','Keywords','Evidence','Tailor Resume','Raw JSON'])
    with tabs[0]:
        st.markdown(a.get('overall_summary',''))
        st.subheader('Recommendations')
        for x in a.get('recommendations',[]): st.write('• '+x)
        st.subheader('ATS Issues')
        for x in a.get('ats_issues',[]): st.write('• '+x)
    with tabs[1]:
        st.subheader('Matched skills')
        for x in a.get('matched_skills',[]): st.markdown(f'**{x.get("skill")}** — {x.get("match_type")}  \n{x.get("reason")}')
        st.subheader('Partial matches')
        for x in a.get('partial_matches',[]): st.markdown(f'**{x.get("jd_requirement")}** — {x.get("gap")}')
        st.subheader('Missing skills')
        for x in a.get('missing_skills',[]): st.markdown(f'**{x.get("skill")}** ({x.get("importance")}) — {x.get("reason")}')
    with tabs[2]:
        st.subheader('Project relevance')
        for x in a.get('project_relevance',[]): st.markdown(f'**{x.get("project")}** — {x.get("relevance")}  \n{x.get("reason")}')
        st.subheader('Experience analysis'); st.write(a.get('experience_analysis',''))
        st.subheader('Education analysis'); st.write(a.get('education_analysis',''))
    with tabs[3]:
        st.write('JD keywords'); st.markdown(' '.join(f'<span class="badge">{x}</span>' for x in a.get('jd_keywords',[])),unsafe_allow_html=True)
        st.write('Missing keywords'); st.markdown(' '.join(f'<span class="badge">{x}</span>' for x in a.get('missing_keywords',[])),unsafe_allow_html=True)
    with tabs[4]:
        for group,label in [('jd','JD retrieval'),('resume','Resume retrieval')]:
            st.subheader(label)
            for i,x in enumerate(result['retrieved'][group],1):
                with st.expander(f'#{i} • score {x.get("rerank_score",0):.4f} • {x["metadata"]["source"]}'):
                    st.write(x['text'])
        st.code('JD → Extraction → Chunking → Embeddings → Chroma → Multi-query Retrieval → Cosine Reranking → Augmentation → LLM → Structured Analysis',language='text')
    with tabs[5]:
        st.write('The generator uses the supplied LaTeX design as the visual source of truth and the RAG analysis as the tailoring source of truth.')
        if st.button('Tailor My Resume',type='primary'):
            try:
                template=Path(__file__).parents[1]/'templates'/'resume_template.tex'
                tex=generate_latex(st.session_state.llm,template.read_text(encoding='utf-8'),result['jd_profile'],result['resume_profile'],result['ats_analysis'])
                out=Path(__file__).parents[1]/'outputs'; out.mkdir(exist_ok=True); tex_path=out/'tailored_resume.tex'; tex_path.write_text(tex,encoding='utf-8')
                comp=compile_latex(tex_path,out); st.session_state.tex=tex; st.session_state.comp=comp
            except Exception as e: st.error(f'Tailoring failed: {e}')
        if st.session_state.get('tex'):
            st.download_button('Download .tex',st.session_state.tex,'tailored_resume.tex','text/plain')
            if st.session_state.comp.get('pdf') and Path(st.session_state.comp['pdf']).exists(): st.download_button('Download PDF',Path(st.session_state.comp['pdf']).read_bytes(),'tailored_resume.pdf','application/pdf')
            else: st.warning(st.session_state.comp.get('message','PDF not compiled.'))
            with st.expander('LaTeX preview'): st.code(st.session_state.tex,language='latex')
    with tabs[6]: st.json(result)
