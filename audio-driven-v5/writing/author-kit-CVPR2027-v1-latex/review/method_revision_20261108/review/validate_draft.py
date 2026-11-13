"""Build-artifact checks for this draft, not scientific-result validation."""
from pathlib import Path
import hashlib,json,re,zipfile
import pymupdf
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent.parent
source=json.loads((P/'review/source.json').read_text())
assert hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['sha256']
texfiles=[P/'main.tex',*sorted((P/'draft_sections').glob('*.tex')),*sorted((P/'tables').glob('*.tex'))]
# The author keeps older drafts as full-line comments; only validate active TeX.
text='\n'.join(re.sub(r'^\s*%.*$', '', x.read_text(), flags=re.M) for x in texfiles)
# Follow actual input order as well as checking individual source files.
# Task and evaluation are now nested; neither should be loaded twice.
loaded=[]
def expand_inputs(path):
 assert path not in loaded, f'Duplicate input: {path}'
 loaded.append(path)
 active=re.sub(r'^\s*%.*$', '', path.read_text(), flags=re.M)
 return re.sub(r'\\input\{([^}]+)\}',lambda m: expand_inputs(P/(m.group(1)+'.tex')),active)
expanded=expand_inputs(P/'main.tex')
section_titles=re.findall(r'\\section\{([^}]+)\}',expanded)
assert section_titles==['Introduction','Related Work','Task and Data Construction','Method','Experiments','Conclusion'],section_titles
section_number=0;subsection_number=0;structure=[]
for match in re.finditer(r'\\(section|subsection)\{([^}]+)\}',expanded):
 level,title=match.groups()
 if level=='section':
  section_number+=1;subsection_number=0;number=str(section_number)
 else:
  subsection_number+=1;number=f'{section_number}.{subsection_number}'
 structure.append({'number':number,'title':title})
assert {'number':'3.1','title':'Task Definition'} in structure
assert {'number':'5.2','title':'Evaluation Protocol'} in structure
assert {'number':'5.5','title':'Ablation Studies'} in structure
labels=re.findall(r'\\label\{([^}]+)\}',expanded)
assert len(labels)==len(set(labels)), 'Duplicate labels'
keys=set()
for cites in re.findall(r'\\cite\w*\{([^}]+)\}',text):keys.update(cites.split(','))
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(P/'references.bib').read_text()))
assert keys==bibkeys,(keys-bibkeys,bibkeys-keys)
figs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
assert len(figs)==5 and all((P/'figures'/x).exists() for x in figs)
for ref in re.findall(r'\\input\{([^}]+)\}',text):assert (P/(ref+'.tex')).exists(),ref
log=(P/'main.log').read_text()
issues=[x for x in log.splitlines() if re.search(r'Overfull|Missing character|(?:LaTeX|Package .*?) Warning:|undefined references|undefined citations',x)]
assert not issues,issues
assert len(re.findall(r'^A[0-4]:',(P/'tables/T3_ablation.tex').read_text(),re.M))==5
assert text.count('\\begin{table*}')==6
quote='评价针对最终生成的音视频进行，不以训练辅助头的分类结果替代生成效果。各项分数同时报告有效样本或窗口覆盖率；未知标签、检测失败与真正无事件分开处理，不将不可测项填为零或满分。'
assert quote in text
assert 'MLLM-Based Cascaded Generation' in text and '本小节尚无写定正文' in text
d=pymupdf.open(P/'main.pdf');out=[];pageinfo=[]
for i,page in enumerate(d):
 t=page.get_text();out.append(t)
 spans=[span for block in page.get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line.get('spans',[])]
 outside=[x['text'] for x in spans if not page.rect.contains(pymupdf.Rect(x['bbox']))]
 assert not outside,(i+1,outside)
 page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25)).save(P/f'review/page_{i+1:02d}.png')
 pageinfo.append({'page':i+1,'text_chars':len(t),'vector_drawings':len(page.get_drawings()),'text_outside_page':outside})
alltext='\n'.join(out);(P/'review/pdf_text.txt').write_text(alltext)
# CVPR review line numbers interrupt extracted prose; remove isolated number lines only.
prose=re.sub(r'^\s*\d+\s*$','',alltext,flags=re.M)
assert re.sub(r'\s','',quote) in re.sub(r'\s','',prose)
for marker in ['704,368','7,087','284,218','MLLM-Based Cascaded Generation','F1:','F2:','F3:','F4:','F5:','T2a:','T2b:','T3:','T4:','References']:
 assert marker in alltext,marker
for start in range(0,len(d),4):
 canvas=Image.new('RGB',(1240,1680),'#dbe0e5');draw=ImageDraw.Draw(canvas)
 for k in range(4):
  i=start+k
  if i>=len(d):break
  img=Image.open(P/f'review/page_{i+1:02d}.png');img.thumbnail((600,800))
  x=10+(k%2)*620;y=24+(k//2)*840
  canvas.paste(img,(x,y));draw.text((x,y-18),f'Page {i+1}',fill='black')
 canvas.save(P/f'review/contact_{start//4+1}.png')
for name in ['T2a_av.tex','T2b_visual.tex','T3_ablation.tex','T4_human.tex']:
 for line in (P/'tables'/name).read_text().splitlines():
  if '\\pending' in line:
   assert not any(re.fullmatch(r'\s*[0-9]+(?:\.[0-9]+)?\s*(?:\\\\)?\s*',c) for c in line.split('&')[1:]),(name,line)
for dest in re.findall(r'\]\(<([^>]+)>\)',(P/'CITATIONS.md').read_text()):
 if dest.startswith('..'):assert (P/dest).exists(),dest
z=zipfile.ZipFile(P.parent/'author-kit-CVPR2027-v1-latex.zip');style_checks={}
for filename in ['cvpr.sty','ieeenat_fullname.bst']:
 candidates=[n for n in z.namelist() if (n==filename or n.endswith('/'+filename)) and '__MACOSX' not in n]
 assert len(candidates)==1,candidates
 assert z.read(candidates[0])==(P/filename).read_bytes(),filename
 style_checks[filename]='matches supplied template archive'
report={'status':'compiled and structurally checked; semantic and visual review recorded separately','pages':len(d),'figures':len(figs),'table_floats':text.count('\\begin{table*}'),'citation_keys':len(keys),'english_prose_paragraphs':sum(len(re.findall(r'^% Paragraph role P\d+', f.read_text(), re.M)) for f in (P/'draft_sections').glob('*.tex')),'ablation_rows':5,'undefined_citations_or_references':False,'latex_warnings_or_overfull_boxes':issues,'chinese_source_unchanged':True,'required_chinese_note_rendered':True,'performance_results_unfilled':True,'original_style':style_checks,'pdf_sha256':hashlib.sha256((P/'main.pdf').read_bytes()).hexdigest(),'page_checks':pageinfo,'review_documents':['含义与证据核对.md','段落反向提纲.md'],'research_skill_installed':Path('/private/liqi_workspace/.codex/skills/research-paper-writing/SKILL.md').exists()}
report['main_sections']=section_titles
report['section_structure']=structure
report['nested_inputs_loaded_once']=True
report['review_documents'].append('行文结构调整_20261108.md')
(P/'review/VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='page_checks'},ensure_ascii=False,indent=2))
