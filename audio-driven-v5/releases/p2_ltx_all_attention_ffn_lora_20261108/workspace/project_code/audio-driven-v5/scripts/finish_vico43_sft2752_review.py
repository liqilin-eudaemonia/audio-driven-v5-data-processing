"""Finish review artifacts after the observed SFT2752 numerical pipeline."""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from vico_avatarforcing import read,save,sha

WORK=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
PAGE=ROOT/'data_previews/listen_benchmark_review/vico43_sft2752_20261107'


def integrity():
    import numpy as np
    import soundfile as sf
    previous=ROOT/'experiments/p2_vico43_unified_20261107'
    frozen=read(previous/'evaluation_protocol.json')
    for group in ['code_and_deca_assets','bfm_assets','fid_weights']:
        for path,h in frozen[group].items():assert sha(path)==h,path
    assert sha(frozen['arcface_path'])==frozen['arcface_sha256']
    save(WORK/'SCORER_REUSE_AUDIT.json',dict(passed=True,
        prior_protocol_sha256=sha(previous/'evaluation_protocol.json'),
        same_code_and_bfm_deca_arcface_fid_assets=True))
    rows=[]
    for item in read(WORK/'shards/00.json'):
        path=WORK/'generated/combined8000_sft2752'/item['key']/'listener.wav'
        wave,rate=sf.read(path,always_2d=True)
        assert np.isfinite(wave).all()
        assert len(wave)==round(item['geometry']['duration_s']*rate)
        rows.append(dict(id=item['id'],frames=len(wave),rate=rate,channels=wave.shape[1],
            rms=float(np.sqrt(np.square(wave).mean())),peak=float(np.abs(wave).max()),sha256=sha(path)))
    save(WORK/'AUDIO_INTEGRITY.json',dict(passed=True,pairs=43,finite_and_duration_checked=True,
        scope='Signal integrity only; RMS is not a speech or quality label',rows=rows))


def documentation():
    result=read(WORK/'COMPARISON.json');url=read(WORK/'REVIEW.json')['url']
    a=result['images']['joint'];b=result['images']['joint_sft2752']
    original=WORK/'README.md';text=original.read_text()
    start=text.index('**当前：');end=text.index('\n\n',start)
    text=text[:start]+f'**已完成：43/43条SFT2752生成、官方画质及BFM／DECA动作评分，数值／媒体／浏览器检查通过。** [全部43条同步视频与评分]({url}) · [完整对比与配对区间](RESULTS.md) · [四模型JSON](COMPARISON.json) · [逐条规划差异](planning_comparison.csv)。原始Qwen与两种baseline封存结果保持不变。\n\nPSNR {a["psnr"]:.5f} → {b["psnr"]:.5f}；SSIM {a["ssim"]:.5f} → {b["ssim"]:.5f}；FID {a["fid"]:.5f} → {b["fid"]:.5f}。动作距离同时结合GT方差和来源聚类区间解释，人工语境适当性评分尚未开展。'+text[end:]
    text+='\n[媒体／独立L1复算](FINAL_NUMERICAL_AUDIT.json) · [浏览器验收](BROWSER_AUDIT.json) · [评分代码及权重复用核验](SCORER_REUSE_AUDIT.json) · [声音完整性](AUDIO_INTEGRITY.json)。首批9条pilot页面为历史中间结果，不代替完整43条表。\n'
    original.write_text(text)
    index=ROOT.parent/'OmniResponse/evaluation/README.md';text=index.read_text()
    prefix='**ViCo43 最新SFT2752替换测评进行中（2026-11-07）：**'
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith(prefix):
            lines[i]=f'**ViCo43 最新SFT2752替换测评已完成（2026-11-07）：** 固定联合8000，只替换Qwen规划器及其描述／H／Gemma；全部43条正式test、Whisper、种子及采样参数不变。[完整新旧视频与评分]({url}) · [结果、协议与验收](../../audio-driven-v5/experiments/p2_vico43_combined8000_sft2752_20261107/README.md)。官方画质、BFM／DECA、配对来源聚类区间及浏览器验收完成；旧结果保持封存。生成器训练H来自base Qwen，本轮推理H来自SFT2752；不把单项距离改善等同于反馈更合适。'
            break
    else:raise AssertionError('Expected SFT evaluation entry missing')
    index.write_text('\n'.join(lines)+'\n')


def finish():
    completed=WORK/'COMPLETE.json'
    if completed.exists():
        result=read(completed);assert result['success']
        for path,h in result['files'].items():assert sha(path)==h,path
        print('Already complete; sealed artifacts verified; no process restarted')
        return
    save(WORK/'REVIEW_FINISH_RUNTIME.json',dict(pid=os.getpid(),at=time.time()))
    deadline=time.monotonic()+4*3600
    marker=WORK/'MOTION_COMPARISON_COMPLETE.json'
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Numerical pipeline incomplete; no evaluation or training restarted')
        time.sleep(10)
    assert read(marker)['passed']
    assert read(WORK/'FINAL_NUMERICAL_AUDIT.json')['passed']
    l2l=ROOT/'compare/.envs/l2l/bin/python'
    subprocess.run([str(l2l),str(ROOT/'scripts/build_vico43_sft2752_review.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/check_vico43_sft2752_browser.py')],cwd=ROOT,check=True)
    assert read(WORK/'BROWSER_AUDIT.json')['passed']
    integrity()
    motion=read(WORK/'motion_protocol.json')
    previous=ROOT/'experiments/p2_vico43_unified_20261107/RESULTS.json'
    assert sha(previous)==motion['sources'][str(previous)]['sha256']
    documentation()
    files=[WORK/n for n in ['protocol.json','motion_protocol.json','COMPARISON.json',
        'IMAGE_RESULTS.json','BFM_RESULTS.json','DECA_RESULTS.json','CONDITION_DIFFERENCE.json',
        'FINAL_NUMERICAL_AUDIT.json','BROWSER_AUDIT.json','planning_comparison.csv',
        'image_comparison.csv','RESULTS.md','REVIEW.json','README.md',
        'SCORER_REUSE_AUDIT.json','AUDIO_INTEGRITY.json']]
    files += [PAGE/'index.html',PAGE/'data.json',Path(__file__),
        ROOT/'scripts/p2_vico43_sft2752_compare.py',ROOT/'scripts/build_vico43_sft2752_review.py',
        ROOT/'scripts/check_vico43_sft2752_browser.py']
    save(WORK/'COMPLETE.json',dict(success=True,pairs=43,planner_step=2752,generator_step=8000,
        at=time.time(),files={str(p):sha(p) for p in files},
        prior_results_unchanged=True,human_scores_collected=0,
        url=read(WORK/'REVIEW.json')['url']))
    save(WORK/'status.json',dict(stage='complete',total=43,at=time.time()))


if __name__=='__main__':
    with (WORK/'review_finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finish()
