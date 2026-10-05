import re, subprocess, shutil
from pathlib import Path
from .prompts import tailor_prompt

def strip_code_fence(text):
    text=text.strip()
    text=re.sub(r'^```(?:latex|tex)?\s*','',text,flags=re.I)
    text=re.sub(r'\s*```$','',text)
    return text.strip()

def generate_latex(llm, template, jd_profile, resume_profile, analysis):
    return strip_code_fence(llm.generate('You are a careful LaTeX resume editor. Return only compilable LaTeX.', tailor_prompt(template,jd_profile,resume_profile,analysis), max_tokens=7000))

def compile_latex(tex_path, output_dir):
    compiler=shutil.which('pdflatex')
    if not compiler: return {'compiled':False,'message':'pdflatex was not found. The .tex file was generated successfully; install a LaTeX distribution such as MiKTeX or TeX Live to compile the PDF.'}
    tex_path=Path(tex_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    cmd=[compiler,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={out}',str(tex_path)]
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90,cwd=tex_path.parent)
        pdf=out/(tex_path.stem+'.pdf')
        return {'compiled':p.returncode==0 and pdf.exists(),'pdf':str(pdf) if pdf.exists() else None,'message':'PDF compiled successfully.' if pdf.exists() else p.stdout[-3000:]}
    except Exception as e: return {'compiled':False,'message':str(e)}
