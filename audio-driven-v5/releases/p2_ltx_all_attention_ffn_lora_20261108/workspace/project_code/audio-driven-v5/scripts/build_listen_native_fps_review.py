#!/usr/bin/env python3
"""Build a lightweight blinded review from existing, immutable 50-clip outputs."""
import difflib
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'experiments/listen_visual_input50_20261029'
DECISION = RUN/'native_fps_decision'
PARENT = ROOT/'data_previews/listen_benchmark_review/prompt_versions_20261029'
OUT = PARENT/'fps_decision'

# These are assistant observations from sampled cached video frames, not labels.
# Questions remain neutral; observations are collapsed by default in the UI.
CASES = {
 'S0157': ('再次微笑是否被保留', [0.7,3.3],
   '笑容减弱之后，约1.8–2.5秒有没有再次明显露齿微笑？',
   ['有再次微笑','没有再次微笑','画面不够明确'],
   '抽帧可见笑容再次增强。video16单列1.9–2.4秒，video8将0.8–2.8秒概括为笑容消退。video16的3.2–8.9秒持续抿嘴仍有重复切分，不能据此判整条全面更好。'),
 'S0333': ('短暂闭眼是否真实', [1.4,3.8],
   '约1.8秒和3.2秒附近，是否各有一次短暂闭眼？',
   ['两处都有','只有一处','两处都没有','画面不够明确'],
   '密集抽帧可见约1.76–1.84秒、3.12–3.44秒的眼部闭合。video16单独描述了两处变化，video8主要概括微笑和转头。抽帧观察不等于精确边界真值。'),
 'S0321': ('张嘴的时间组织', [1.8,3.1],
   '张嘴是否真实发生；单列时间区间是否比写在长描述内更好用？',
   ['动作真实，单列区间更好','动作真实，两种表达都可','所写动作或时间不正确','无法判断'],
   'video8已在长描述中提到约2.3秒和8.8秒张嘴；video16单列区间。已抽帧确认前一处约2.32–2.56秒张嘴，本项主要检验时间组织，不应算video16新增发现两个动作；末尾一处可播放整段补查。'),
 'S1833': ('一次眨眼是否被拆成两次', [0,1.1],
   '开始0.7秒内，有几次完整的闭眼—睁眼过程？',
   ['一次','两次','没有','无法判断'],
   '20fps抽帧呈现一段连续闭眼—睁眼过程，未见两个完整周期。video16写成0–0.4和0.4–0.7秒两次眨眼，疑似误拆；video8的笼统眼部变化也不等于完整标注。'),
 'S1033': ('密集循环是否有画面依据', [0,7.168],
   '反复出现的张嘴／闭嘴、向上／向下看，是否对应同样多次真实变化？',
   ['主要是重复切碎','多数对应真实多次变化','重复与真实变化混合','无法判断'],
   'video16有71条，约每0.1秒交替两句话；video8也有15条重复。抽帧主要呈现较持续的面部状态，末尾张嘴更明显，未支持按条目数理解为独立动作。4fps抽帧不能逐一排除所有瞬时变化，建议慢放复核。'),
 'S2147': ('持续状态是否被反复标记', [0,4.821],
   '这些连续相邻区间主要是在重复描述持续状态，还是不同的独立动作？',
   ['主要是持续状态重复','主要是独立动作','两者都有','无法判断'],
   'video16有24条，约每0.2秒重复睁眼及嘴开合状态，后半段大量同句；抽帧显示持续状态中夹杂眨眼与后段嘴部变化，不支持把每条都算新动作。video8也需核验具体描述。'),
 'S1052': ('持续微笑与反复开合嘴', [0,2.94],
   '张嘴之后主要是持续微笑，还是反复进行了多次张嘴—闭嘴？',
   ['主要是一次张嘴后持续微笑','确有多次张嘴—闭嘴','两种描述都不准确','无法判断'],
   '8fps抽帧显示约0.4秒张嘴，约1秒后嘴部收拢并持续微笑。video8的11条多次开合描述疑似切碎／误报，video16的4条较贴近主要过程；细微嘴部变化仍不能简单当作完全静止。'),
 'S2128': ('重复凝视描述是否需要多条', [0,6.704],
   '开头变化后，是否主要保持抿嘴与注视状态？',
   ['主要保持持续状态','后面还有多次明显独立动作','两种概括都遗漏重要变化','无法判断'],
   'video8有27条，其中23条重复描述向上／右看和噘嘴；video16有3条。抽帧后段以持续抿嘴和注视为主，支持合并持续状态，但仍有眨眼和轻微姿态变化，不能把3条直接当作完整真值。'),
 'S0343': ('规律抬头低头是否真实', [0,4.99],
   '是否确实约每0.5秒交替抬头和低头，形成多次完整周期？',
   ['确有这样多次周期','没有，主要是较连续姿态变化','有重复动作但次数不符','无法判断'],
   'video16列出10个约0.5秒交替抬／低头区间。抽帧可见缓慢姿态变化、后段向一侧转动，未支持如此规律的五次上下周期。需视频慢放确认，不能把规律时间表当作动作召回。'),
}

RANDOM_NOTES = {
 'S1236':'4fps抽帧中，开头面部状态较稳定，但“目光右移再回中”不易仅凭抽帧确认。约2.3秒伴随微笑有闭眼，两组主要写微笑／眼角变化，可能共同漏掉短暂闭眼的独立区间。此项仍保留在随机组，不因发现问题重新抽样。',
 'S0284':'抽帧支持开头向上看、嘴略张的共同描述。整段还可见随后笑容增强、低头再抬头，末段约7.5秒再次张嘴；请检查两组对末段嘴部变化的持续时间是否过宽。',
 'S1668':'抽帧支持后段头部较稳定、嘴略张，但“眼神接触摄像机”及“正在说话”不能仅由这一画面确定。开头确有从侧向朝较正面转动，两组方向描述不一致；“相近项”不代表整段所有描述都一致。',
 'S0339':'抽帧可见嘴部形状改变、头部与眉部变化；两组整段文字完全相同。它们都笼统写speaks和偶尔眨眼，没有给出细分变化时间。抽帧不足以确认具体眨眼次数，也不能只靠视觉证实发声。',
 'S0694':'抽帧中约1.2秒后明显露齿微笑并持续，期间有嘴部变化。两组都用“嘴开合，如同说话”概括，可能共同漏掉微笑状态；不要因为两组同句就默认正确。',
 'S0127':'抽帧支持开头嘴略张及随后闭嘴。整段末尾约6.28秒可见再次张嘴，video8有对应条目，video16称其余时段一直闭嘴、无可见运动，疑似遗漏。随机项本身在开头，需按要求播放整段才能发现此差异。',
}


def read(p):return json.loads(p.read_text())
def normalize(s):return ' '.join(s.lower().split())


def main():
    OUT.mkdir(parents=True,exist_ok=True);DECISION.mkdir(parents=True,exist_ok=True)
    previous=read(PARENT/'data.json')
    originals={s['id']:s for s in previous['samples']}
    sources={}; samples=[]; agreement_pool=[]
    order=list(originals);random.Random(20261030).shuffle(order)
    for sid in sorted(originals):
        path=RUN/'results'/(sid+'.json');raw=read(path);old=originals[sid]
        sources[sid]=hashlib.sha256(path.read_bytes()).hexdigest()
        series={a:[{k:r.get(k) for k in ('view','start','end','status','parsed','raw','prompt','finish_reason','error')}
                   for r in raw['records'] if r['arm']==a] for a in ('video8','video16')}
        fs={a:next(r for r in series[a] if r['view']=='full')['parsed']['events'] for a in series}
        # Reproducible textual-nearness screen, deliberately not semantic matching.
        pairs=[]
        for i,a in enumerate(fs['video8']):
            for j,b in enumerate(fs['video16']):
                sim=difflib.SequenceMatcher(None,normalize(a['behavior']),normalize(b['behavior'])).ratio()
                delta=max(abs(a['start']-b['start']),abs(a['end']-b['end']))
                if sim>=.90 and delta<=.200001:
                    c={'sid':sid,'indices':[i,j],'similarity':sim,'max_boundary_delta_s':delta,'events':[a,b]}
                    pairs.append(c);agreement_pool.append(c)
        counts={a:{'entries':len(es),'exact_repeat_extras':len(es)-len(set(normalize(e['behavior']) for e in es))} for a,es in fs.items()}
        item={k:old[k] for k in ('id','short','duration','render_duration','source_start_s','crop')}
        item.update(video='../'+old['video'],audio=dict(old['audio']),series=series,counts=counts,
                    assignment=['video8','video16'] if order.index(sid)%2==0 else ['video16','video8'],
                    text_near_pairs=pairs,source_sha256=sources[sid])
        for k in ('original','speaker'):item['audio'][k]='../'+old['audio'][k]
        item['task']={'id':'all-'+sid,'group':'all','title':'自由复核','question':'哪份描述更符合实际动作与时间？',
                      'options':['两份基本都正确','一份更准确，请在下方选出','两份都有明显问题','无法判断'],
                      'range':[0,item['duration']],'candidate':'未作单独画面判定。请先看视频，再比较两份原文。'}
        if item['short'] in CASES:
            title,span,question,options,candidate=CASES[item['short']]
            item['task']={'id':'key-'+sid,'group':'key','title':title,'question':question,'options':options,
                          'range':[span[0],min(span[1],item['duration'])],'candidate':candidate,
                          'evidence':str((RUN/'qualitative_review'/(item['short']+'.json')).relative_to(ROOT))}
        samples.append(item)
    # Sample uniformly over eligible clips, then uniformly over their near pairs.
    eligible=sorted(set(x['sid'] for x in agreement_pool if originals[x['sid']]['short'] not in CASES))
    rng=random.Random(20261030)
    random_ids=rng.sample(eligible,min(6,len(eligible)))
    random_selected=[]
    for sid in random_ids:
        pair=rng.choice([x for x in agreement_pool if x['sid']==sid]);random_selected.append(pair)
        s=next(s for s in samples if s['id']==sid)
        a,b=pair['events'];span=[max(0,min(a['start'],b['start'])-.3),min(s['duration'],max(a['end'],b['end'])+.3)]
        s['task']={'id':'random-'+sid,'group':'random','title':'随机一致项抽查','range':span,
                   'question':'两份相近描述是否都符合画面？请再看整段，留意双方是否共同漏掉动作。',
                   'options':['相近描述正确，未发现明显共同遗漏','相近描述正确，但双方共同漏标','两份相近描述都存在错误','其实两份含义不同','无法判断'],
                   'matched_indices':pair['indices'],'candidate':'由文本相似度≥0.90、起止差≤0.2秒筛出，固定随机种子抽取；相近不代表正确。先播放对应区间，再播放整段检查共同遗漏。 '+RANDOM_NOTES.get(s['short'],''),
                   'evidence':str((DECISION/'frames'/(s['short']+'.json')).relative_to(ROOT)),
                   'similarity':pair['similarity'],'max_boundary_delta_s':pair['max_boundary_delta_s']}
    tasks=[s['id'] for code in CASES for s in samples if s['short']==code]+random_ids
    payload={'schema':'listen-native-fps-light-review-v1','run_id':'native_fps_decision_20261029',
             'human_ground_truth':False,'training_admitted':False,'samples':samples,'queue':tasks,
             'sampling':{'seed':20261030,'eligible_clip_count':len(eligible),'random_ids':random_ids,
                         'rule':'text ratio >=.90 and both endpoints within .2s; sample clips then event pairs; exclude key-review clips'},
             'scope':'15 targeted/random review tasks plus all50 raw outputs; not a representative accuracy/recall estimate',
             'quality_findings_are':'assistant sampled-frame observations, pending user confirmation'}
    if len(tasks)!=15 or len(set(tasks))!=15:raise ValueError('Expected nine key plus six distinct random tasks')
    (OUT/'data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
    for name in ('listen_native_fps_review.html','listen_native_fps_review.js'):
        shutil.copyfile(ROOT/'scripts'/name,OUT/('index.html' if name.endswith('.html') else 'review.js'))
    shutil.copyfile(ROOT/'scripts/listen_visual_prompt_review_audio.js',OUT/'audio.js')
    speed=OUT/'speed'
    if not speed.exists():speed.symlink_to(DECISION/'speed',target_is_directory=True)
    audit={'samples':50,'key_tasks':9,'random_tasks':6,'descriptions_unmodified':True,
           'source_sha256':sources,'random_selected':random_selected,'eligible_random_clips':eligible,
           'human_ground_truth':False,'training_admitted':False,'no_labels_modified':True}
    (DECISION/'review_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (DECISION/'review_cases.json').write_text(json.dumps({s['id']:s['task'] for s in samples if s['id'] in tasks},ensure_ascii=False,indent=2)+'\n')
    marker='id="native-fps-decision-link"'
    banner='<p '+marker+'><a href="fps_decision/">新增：8／16 fps 关键分歧复核＋随机一致项抽查＋独立测速</a></p>'
    # Template and already-published old page both retain existing modes and state.
    for path in (ROOT/'scripts/listen_visual_prompt_versions_review.html',PARENT/'index.html'):
        text=path.read_text()
        if marker not in text:path.write_text(text.replace('</header>',banner+'</header>',1))
    print(json.dumps({'page':str(OUT),'samples':50,'key':9,'random':6,'random_ids':random_ids},ensure_ascii=False))


if __name__=='__main__':main()
