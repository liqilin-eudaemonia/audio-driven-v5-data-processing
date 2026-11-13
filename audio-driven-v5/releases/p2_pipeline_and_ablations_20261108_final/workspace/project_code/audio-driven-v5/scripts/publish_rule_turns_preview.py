"""Build the v2 review page, with full-conversation audio and floor provenance."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_previews/seamless_rule_turns_v2'
V1=ROOT/'data_previews/seamless_face_listening_v1'


def replace(text,a,b):
    if a not in text:raise ValueError('Missing template anchor: '+a[:90])
    return text.replace(a,b)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    t=(ROOT/'scripts/seamless_preview_template.html').read_text()
    t=replace(t,'Seamless · 倾听片段审阅','Seamless · 规则话轮 V2')
    t=replace(t,'SEAMLESS INTERACTION / REVIEW 01','SEAMLESS INTERACTION / RULE TURNS 02')
    t=replace(t,'倾听片段 · 脸部音视频审阅','规则话轮 V2 · 倾听音视频')
    t=replace(t,'对比双方画面、切换音轨，检查话轮边界与固定裁剪窗口。','停顿不单独切断话轮 · 插话独立标记 · 完整会话上下文 · 512×512')
    t=replace(t,'value="short_feedback_candidate"','value="feedback_candidate"')
    t=replace(t,'有 Listener 短发言（待分类）','有规则反馈候选')
    t=replace(t,"short_feedback_candidate:'短发言待分类'","feedback_candidate:'规则反馈候选'")
    old_notice='本页保留第一版分段，脸部已更新为每人 512×512。短发言尚未区分倾听反馈、插话和正式接话；话轮规则待讨论后更新。“较安静”也只表示现有转写/VAD 未检出发声。保留静音，不对单条音轨做响度归一化。'
    new_notice='本版已重做话轮切分，仅使用现有转写、VAD 和明确规则，没有运行标注模型。主话轮允许停顿；插话、短问句及不确定发言从倾听导出区间中排除，并保留原时间。以下仍是规则候选，请用完整会话检查接管边界。'
    t=replace(t,old_notice,new_notice)
    t=replace(t,'Listener 转写 / 短发言待分类','Listener 转写 / 规则保留的声音事件')
    t=replace(t,"'seamless_review_v1'","'seamless_rule_turns_v2'")
    t=replace(t,"'疑似插话','倾听反馈'","'漏切正式接话','话轮切早了','反馈被误排除','疑似插话','倾听反馈'")
    t=replace(t,'固定随机种子 20261021 · test split','规则话轮 V2 · 原 50 个会话 · test split')
    t=replace(t,'<a href="manifest.json">完整样本清单</a>','<a href="manifest.json">完整样本清单</a> · <a href="../index_v1.html">第一版对照</a>')
    t=replace(t,'pairNo:pi+1,clipNo:ci+1,key:p.pair+\'/\'+ci',
              "pairNo:pi+1,clipNo:ci+1,conversation_path:p.conversation_path,conversation_roles:p.conversation_roles,key:p.pair+'/'+c.turn_id+'/'+c.part_index")
    t=replace(t,'<section class="panel player">','<section class="panel"><div class="label" id="turnInfo"></div><div class="caption" id="turnReason"></div><svg id="floorTimeline" viewBox="0 0 850 60" style="width:100%;height:60px"></svg><div class="caption">绿色：当前导出区间；橙色：因 listener 发言排除的区间。下面时间轴仍显示当前片段的 VAD。</div></section><section class="panel player">')
    conversation_html='''<details class="panel" id="conversationDetails"><summary>完整会话与切点上下文（含双方声音）</summary><div class="caption" id="conversationRoles"></div><div class="toolbar"><button id="contextStart">播放起点前后各 5 秒</button><button id="contextEnd">播放终点前后各 5 秒</button><button id="contextFloor">播放完整主话轮</button><button id="contextFree">自由播放会话</button></div><video id="conversation" controls playsinline preload="none"></video><div class="caption" id="conversationClock"></div><div id="parentEvents"></div></details>'''
    t=replace(t,'<details class="panel" id="contextDetails">',conversation_html+'<details class="panel" id="contextDetails">')
    t=replace(t,'renderReview();renderList();updatePosition();','renderReview();renderList();updatePosition();showTurnContext();')
    t=replace(t,'select(filtered[0]);}', 'select(filtered[0]);updatePosition();}')
    t=replace(t,"$('subtitle').textContent=`原会话 ${fmt(c.start)}–${fmt(c.end)} · ${c.duration.toFixed(1)} 秒 · ${labels[c.kind]}`;",
              "$('subtitle').textContent=`${c.turn_id} · ${c.entire_floor_turn?'完整主话轮':'剔除插话后的连续倾听子区间'} · 原会话 ${fmt(c.start)}–${fmt(c.end)} · ${c.duration.toFixed(1)} 秒 · ${labels[c.kind]}`;")
    t=replace(t,"start:c.start,end:c.end,speaker_vad_fraction", "start:c.start,end:c.end,parent_turn:[c.parent_turn_start,c.parent_turn_end],start_reason:c.start_reason,end_reason:c.end_reason,boundary_evidence:c.boundary_evidence,speaker_vad_fraction")
    t=replace(t,"<span class=\"num\">${String(c.pairNo).padStart(2,'0')} · 片段 ${c.clipNo}</span>","<span class=\"num\">${String(c.pairNo).padStart(2,'0')} · ${c.turn_id}.${c.part_index}</span>")
    t=replace(t,"end:c.end}:{})", "end:c.end,turn_id:c.turn_id,segmentation_revision:2}:{})")
    # Additional full-conversation player is intentionally independent of the crop player.
    extra=r'''
const conversation=$('conversation');let stopAt=null;
const eventLabels={feedback_candidate:'反馈候选',interruption_candidate:'插话候选',clarification_candidate:'澄清问句候选',uncertain_speech:'不确定发言',overlap_from_previous_turn:'上一话轮延续的重叠语音',untranscribed_vocalization:'无转写的发声'};
const reasons={first_substantive_utterance:'首次实质发言',rule_takeover:'对方发言满足规则接管条件',other_person_takeover_onset:'在对方接管发言的起点结束',conversation_end:'到达会话结束',after_excluded_listener_event:'从被排除的 listener 事件之后开始',before_excluded_listener_event:'在被排除的 listener 事件之前结束'};
function showTurnContext(){
 const c=current;conversation.pause();stopAt=null;
 const url=new URL(c.conversation_path,location.href).href;
 if(conversation.src!==url){conversation.src=c.conversation_path;conversation.load();}
 $('turnInfo').textContent=`所属主话轮 ${c.turn_id}：${fmt(c.parent_turn_start)}–${fmt(c.parent_turn_end)}（${(c.parent_turn_end-c.parent_turn_start).toFixed(1)} 秒）`;
 $('turnReason').textContent=`当前导出：${fmt(c.start)}–${fmt(c.end)}。起点：${reasons[c.start_reason]||c.start_reason}；终点：${reasons[c.end_reason]||c.end_reason}。${Object.values(c.crop_boxes).some(b=>b.wide_crop)?'该话轮头部位移较大，固定裁剪框较宽。':''}`;
 const pa=c.parent_turn_start,pb=c.parent_turn_end,w=pb-pa;const xpos=t=>20+(t-pa)/w*810;
 let h=`<rect x="20" y="15" width="810" height="18" rx="4" fill="#e9eee7"/>`;
 c.parent_events.filter(e=>e.exclude).forEach(e=>{h+=`<rect x="${xpos(e.start)}" y="15" width="${Math.max(1,(e.end-e.start)/w*810)}" height="18" fill="#d79a58"/>`;});
 h+=`<rect x="${xpos(c.start)}" y="15" width="${(c.end-c.start)/w*810}" height="18" fill="#19745f"/><text x="20" y="53" font-size="11" fill="#687770">${fmt(pa)}</text><text x="780" y="53" font-size="11" fill="#687770">${fmt(pb)}</text>`;$('floorTimeline').innerHTML=h;
 $('conversationRoles').textContent='完整会话固定布局：'+c.conversation_roles.map((id,i)=>`${i===0?'左':'右'} ${id.split('_').at(-1)}（当前${id===c.speaker?' speaker':' listener'}）`).join('；');
 $('parentEvents').innerHTML='<div class="label" style="margin-top:16px">本主话轮内的 listener 事件</div>'+c.parent_events.map((e,i)=>`<div style="padding:6px 0;border-top:1px solid #e5e9e2"><button data-event="${i}" style="font-size:11px">${fmt(e.start)}–${fmt(e.end)}</button> <span class="pill ${e.exclude?'voice':''}">${eventLabels[e.label]||e.label} · ${e.exclude?'排除':'保留候选'}</span> ${esc(e.text)}</div>`).join('');
 $('parentEvents').querySelectorAll('[data-event]').forEach(el=>el.onclick=()=>{const e=current.parent_events[Number(el.dataset.event)];playContext(Math.max(0,e.start-5),Math.min(conversation.duration||Infinity,e.end+5));});
}
function playContext(a,b){video.pause();audio.pause();context.pause();$('conversationDetails').open=true;stopAt=b;const start=()=>{conversation.currentTime=a;conversation.play().catch(()=>{});};if(conversation.readyState>=1)start();else{conversation.addEventListener('loadedmetadata',start,{once:true});conversation.load();}}
$('contextStart').onclick=()=>playContext(Math.max(0,current.start-5),current.start+5);
$('contextEnd').onclick=()=>playContext(Math.max(0,current.end-5),current.end+5);
$('contextFloor').onclick=()=>playContext(current.parent_turn_start,current.parent_turn_end);
$('contextFree').onclick=()=>{video.pause();stopAt=null;conversation.play().catch(()=>{});};
conversation.addEventListener('play',()=>{video.pause();audio.pause();context.pause();});
video.addEventListener('play',()=>conversation.pause());
conversation.addEventListener('timeupdate',()=>{$('conversationClock').textContent=`原会话时间 ${fmt(conversation.currentTime)} / ${fmt(conversation.duration||0)}`;if(stopAt!==null&&conversation.currentTime>=stopAt){conversation.pause();stopAt=null;}});
'''
    t=replace(t,"$('next').onclick=()=>step(1);filter();","$('next').onclick=()=>step(1);"+extra+'\nfilter();')
    t=replace(t,'const DATA=__MANIFEST__;','const DATA='+json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')+';')
    (OUT/'index.html').write_text(t)
    (OUT/'assets').mkdir(exist_ok=True)
    shutil.copy2(V1/'assets/font-license.txt',OUT/'assets/font-license.txt')
    # Include the new UI's characters in the local Chinese font subset.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=t);sub.subset(font);font.flavor='woff2';font.save(OUT/'assets/review-chinese.woff2')
    link=V1/'rule_turns'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print('Published',len(manifest['pairs']),'pairs',sum(len(p['clips']) for p in manifest['pairs']),'rule-based segments')


if __name__=='__main__':main()
