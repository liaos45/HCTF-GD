"""Read-only submission evidence audit. Writes ONLY inside this audit directory."""
from pathlib import Path
import sys,os,re,json,csv,hashlib,time,datetime,subprocess,ast
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MAIN=ROOT/'paper/hctf_visual_informatics_20261001/hctf_gd_visual_informatics.tex'
STATUSES={'VERIFIED','CODE-VERIFIED ONLY','ARCHIVED ONLY','CONTRADICTED','UNVERIFIED'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,text):
    p=(HERE/name).resolve();assert p.is_relative_to(HERE.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

def save(name,obj):write(name,json.dumps(obj,indent=2,ensure_ascii=False,default=str))
def csvout(name,rows):
    p=(HERE/name).resolve();assert p.is_relative_to(HERE.resolve());p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
def text(path):return Path(path).read_text(encoding='utf-8-sig',errors='replace')
def rel(path):
    p=Path(path).resolve()
    return str(p.relative_to(ROOT)).replace('\\','/') if p.is_relative_to(ROOT) else str(p)
def lines(path,start=1,end=None):
    return '\n'.join(f'{i}: {s}' for i,s in enumerate(text(path).splitlines(),1) if i>=start and (end is None or i<=end))
def table(z):
    z=pd.DataFrame(z)
    def fmt(v):return 'NA' if pd.isna(v) else str(v).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(z.columns)+' |','| '+' | '.join(['---']*len(z.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in z.itertuples(index=False,name=None)])
def event(stage,message,command=None):
    p=HERE/'EXECUTION_LOG.md';p.parent.mkdir(exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(f'\n- {datetime.datetime.now(datetime.timezone.utc).isoformat()} | {stage} | {message}'+(f' | `{command}`' if command else '')+'\n')
def read_csv(path,**kw):return pd.read_csv(path,**kw)

def freeze():
    target=HERE/'SOURCE_AND_CHECKPOINT_MANIFEST.csv'
    if target.exists():return
    files=set()
    for directory in [ROOT/'paper',ROOT/'HCTF_GD_FINAL_FIGURES',ROOT/'deepgd']:
        files.update(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in {'.py','.tex','.bib','.pdf','.fls','.sty','.cls'})
    files.update((ROOT/'scripts').glob('*hctf*.py'));files.update((ROOT/'scripts').glob('*community*.py'));files.update((ROOT/'scripts').glob('*caid*.py'));files.update((ROOT/'scripts').glob('*cpsg*.py'))
    files.update(p for p in ROOT.glob('*.py') if any(s in p.name for s in ['train_cser','evaluate_extra','spectral']))
    for name in ['hctf_submission_main','hctf_strict_size_ood','hctf_adaptive_final','hctf_objective_ablations','hctf_submission_extensions']:
        files.update((ROOT/'checkpoints'/name).rglob('*_best.pt'))
    files.update((ROOT/'checkpoints').glob('*hctf*_best.pt'));files.update((ROOT/'checkpoints/hctf_readability_targeted').glob('*_best.pt'))
    rows=[]
    for p in sorted(files):
        if p.is_file():rows.append(dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,role='checkpoint' if p.suffix=='.pt' else 'manuscript_or_code',freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    csvout(target.name,rows)
    save('AUDIT_SCOPE_LOCK.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main=rel(MAIN),main_sha256=sha(MAIN),training=False,new_Test_inference=False,read_only_archived_statistics=True,protected_sealed_holdouts_excluded=True,write_root=str(HERE),later_operators_excluded=['B','AB','S05','L2'],files_frozen=len(rows)))
    event('freeze',f'{len(rows)} existing manuscript/code/checkpoint files hashed before numerical reproduction',f'{sys.executable} HCTF_GD_SUBMISSION_EVIDENCE_AUDIT/audit_common.py')

if __name__=='__main__':freeze();print('FROZEN',len(pd.read_csv(HERE/'SOURCE_AND_CHECKPOINT_MANIFEST.csv')))
