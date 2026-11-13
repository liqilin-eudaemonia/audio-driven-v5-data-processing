from pathlib import Path
import sqlite3,json,csv
W=Path(__file__).resolve().parent;R=W.parents[1]
load=lambda n:json.loads((W/(n+'.json')).read_text())
meta=load('metadata');fer=load('fer');react=load('reactions');categories=load('source_categories')
with sqlite3.connect('file:'+str(R/'evaluation/glare_pair_review/catalog.sqlite')+'?mode=ro',uri=True) as c:rows=c.execute('SELECT dataset,id,frames FROM clips').fetchall()
with sqlite3.connect('file:'+str(W/'fer_stats.sqlite')+'?mode=ro',uri=True) as c:complete={(d,i) for d,i in c.execute('SELECT dataset,id FROM fer WHERE rows=reaction_frames AND consecutive_zero_based=1')}
with sqlite3.connect('file:'+str(W/'reaction_stats.sqlite')+'?mode=ro',uri=True) as c:peaks={(d,i):v for d,i,v in c.execute('SELECT dataset,id,max(max_score) FROM reaction GROUP BY dataset,id')}
checks=[]
def add(name,predicate,note):
 keep=[r for r in rows if predicate(r)];checks.append(dict(rule=name,clips=len(keep),hours=sum(r[2] for r in keep)/90000,note=note))
add('All distinct released reaction clip IDs',lambda r:True,'Each clip counted once across six categories')
add('FER complete and consecutive',lambda r:(r[0],r[1]) in complete,'Every annotation frame has its FER row 0..N-1; MP, source audio and manual checks are additional conditions')
add('Any positive reaction',lambda r:peaks[(r[0],r[1])]>0,'Diagnostic hypothesis; public HDF5 builder does not reject all-zero CSVs')
add('Any reaction > 0.5',lambda r:peaks[(r[0],r[1])]>.5,'Diagnostic hypothesis; not final manually verified events')
add('At least 5 seconds',lambda r:r[2]>=125,'Diagnostic simple duration rule')
add('FER complete and at least 5 seconds',lambda r:(r[0],r[1]) in complete and r[2]>=125,'Combined diagnostic rule')
add('Direct README 10sec audio training eligibility',lambda r:r[2]//60>=7,'385 total frames = 250 current + 10 previous + 125 reference; 7 complete 60-frame chunks. Assumes one released clip maps to one HDF5 clip without padding or merging.')
add('FER complete and direct README training eligibility',lambda r:(r[0],r[1]) in complete and r[2]//60>=7,'Same direct-reconstruction assumptions')
report=dict(paper=dict(pairs=107149,approx_hours=147,reaction_events=64557,train_test_ratio='9:1',source_pdf=str(R/'related/00_GLARE_local.pdf'),pages=[3,4,7,13,14,16]),checks=checks,fer=fer,source_categories=categories,face_validation_sample=load('face_flags_seamless_result'),public_pipeline_remaining_conditions=['MediaPipe all-frame detections','intersection with source HDF5 audio/motion clip and chunk keys','manual face and audio quality checks','manual verification of reaction events'],judgment='Release is a collection of intermediate automatic annotation outputs with partially filtered clip sets. It cannot currently be identified with the final 107149-pair benchmark. The final subset may be contained in this pool, but that inclusion and its IDs are not established. No tested simple rule reproduces both the pair count and total duration.',not_established=['Exact final sample membership','Whether paper test IDs equal the code val IDs','Exact manually corrected reaction event annotations'],sources={'hf_readme':'https://huggingface.co/datasets/lzk901372/GLARE_reaction_annotation_results/blob/main/README.md','usable_filter':'https://github.com/lzk901372/visual_reaction_annotation/blob/main/generate_usable.py','hdf5_injection':'https://github.com/lzk901372/visual_reaction_annotation/blob/main/build_reaction_pruned_h5.py','training_window':'https://github.com/lzk901372/glare/blob/main/train_multigpu.py','chunk_requirement':'https://github.com/lzk901372/glare/blob/main/float/data/base.py','split':'https://github.com/lzk901372/glare/blob/main/float/data/VideoDataset.py'})
(W/'judgment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (W/'filter_comparison.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['rule','clips','hours','note']);writer.writeheader();writer.writerows(checks)
print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
