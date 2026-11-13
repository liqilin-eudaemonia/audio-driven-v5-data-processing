"""Publish a new blind packet using regenerated6000 outputs and upright references."""
import sys,subprocess,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,rows
BASE=ROOT/'experiments/seamless_upright_eval_fix_20261112';GEN=BASE/'generation6000';EVENT=BASE/'event_presence';OLD=ROOT/'experiments/p2_event_presence_pilot_20261108'
if __name__=='__main__':
    assert read(GEN/'COMPLETE.json')['success']
    p=read(GEN/'protocol.json');refs={r['id']:r for r in rows(BASE/'prepared_200/references.jsonl')};ids={r['id'] for r in read(GEN/'shared_items.json')}
    pairs=rows(OLD/'pilot_input.jsonl')
    for r in pairs:
        if r['dataset']=='seamless':
            r['gt_reference']=refs[r['id']]['gt_reference']
            if r['id'] in ids:
                folder=GEN/'generated'/p['arms'][0]['name']/'seamless'/r['id'];r['generated']=str(folder/'generated.mp4');r['generated_audio']=str(folder/'listener.wav')
    source=BASE/'pilot_input.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    from p2_event_presence.media import bind
    if not (EVENT/'contract.json').exists():bind([source],EVENT,BASE/'event_config.json')
    cfg=read(BASE/'event_config.json');import os
    env=dict(os.environ,PYTHONPATH=cfg['media_python_path'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    subprocess.run([cfg['media_python'],str(ROOT/'scripts/p2_event_presence.py'),'prepare','--work',str(EVENT)],env=env,check=True)
    from p2_event_presence.reporting import review
    review(EVENT)
    entries=rows(EVENT/'blind_review_media.jsonl');assert len(entries)==8
    prepared=rows(EVENT/'prepared_pairs.jsonl');assert all(not m['identity']['errors'] for r in prepared for m in r['prepared_media'].values())
    oldweb=ROOT/'data_previews/listen_benchmark_review/event_presence_pilot_20261108';newweb=oldweb.parent/'event_presence_upright_20261112'
    subprocess.run([sys.executable,str(ROOT/'scripts/p2_event_presence_review_packet.py'),'--work',str(EVENT),'--output',str(newweb),'--previous-packet',str(oldweb)],check=True)
    for mod in ['visual','audio']:
        name=f'review_{mod}.html';archive=oldweb/f'review_{mod}_legacy.html'
        if not archive.exists():shutil.copyfile(oldweb/name,archive)
        dest=f'../event_presence_upright_20261112/{name}'
        (oldweb/name).write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={dest}"><title>已切换转正素材</title><p>本核验已迁移到转正版本。<a href="{dest}">打开新版核验</a>；<a href="{archive.name}">旧版标注导出</a>。浏览器原标注保留，新版只沿用素材身份完全相同的标注。</p></html>')
        page=newweb/name;s=page.read_text();s=s.replace('<div class="workspace">','<p class="muted">本页已使用转正来源和重新生成的结果。更换素材的项目需重新核验；<a href="../event_presence_pilot_20261108/review_'+mod+'_legacy.html">旧版标注可在此导出留档</a>。</p><div class="workspace">');page.write_text(s)
    save(BASE/'REVIEW_PUBLISHED.json',dict(media=8,new_page=str(newweb),previous_packet=str(oldweb),changed_media_require_new_labels=True,automatic_detection='pending'))
