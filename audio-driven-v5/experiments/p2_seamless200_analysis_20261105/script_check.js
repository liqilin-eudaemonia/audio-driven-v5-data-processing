
'use strict';
const $=id=>document.getElementById(id), videos={gt:$('videoGT'),a:$('videoA'),b:$('videoB')};
let D=null,C=null,ctx=null,playing=false,mode='listener',focus='gt',sources=[],buffers={},loadVersion=0,changing=false;
const pcmCache=new Map(), storage='seamless200-review-20261105';
const fmt=(v,n=3)=>v==null||!Number.isFinite(v)?'N/A':v.toFixed(n), pct=v=>v==null?'N/A':(v*100).toFixed(1)+'%';
const fail=e=>{$('error').textContent=String(e.message||e);console.error(e);};
function stopAudio(){for(const s of sources){try{s.stop();s.disconnect();}catch(e){}}sources=[];}
function pause(){playing=false;for(const v of Object.values(videos))v.pause();stopAudio();$('play').textContent='▶ 同步播放';}
async function getPCM(meta){
 if(pcmCache.has(meta.url))return pcmCache.get(meta.url);
 if(!ctx)ctx=new AudioContext({sampleRate:16000});
 const r=await fetch(meta.url);if(!r.ok)throw Error('音轨读取失败：'+r.status+' '+meta.url);
 const raw=await r.arrayBuffer();if(raw.byteLength!==meta.samples*4)throw Error('PCM样本长度不匹配');
 if(crypto.subtle){const h=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),x=>x.toString(16).padStart(2,'0')).join('');if(h!==meta.sha256)throw Error('PCM校验失败');}
 const buffer=ctx.createBuffer(1,meta.samples,meta.rate);buffer.copyToChannel(new Float32Array(raw),0);pcmCache.set(meta.url,buffer);return buffer;
}
function audioNow(){
 stopAudio();if(!playing||!ctx)return;
 const at=videos.gt.currentTime, speed=Number($('speed').value),selected=focus==='gt'?buffers.gt:buffers[focus];
 const values=mode==='speaker'?[buffers.speaker]:mode==='both'?[selected,buffers.speaker]:[selected];
 for(const buffer of values){if(!buffer||at>=buffer.duration)continue;const source=ctx.createBufferSource(),gain=ctx.createGain();source.buffer=buffer;source.playbackRate.value=speed;gain.gain.value=mode==='both'?.5:1;source.connect(gain);gain.connect(ctx.destination);source.start(0,at);sources.push(source);}
 $('audioStatus').textContent='当前听：'+(mode==='speaker'?'speaker':focus.toUpperCase()+(mode==='both'?'＋speaker':''));
}
async function play(){if(changing)return;await ctx.resume();if(videos.gt.currentTime>=C.duration-.04)seek(0);const at=videos.gt.currentTime;for(const v of Object.values(videos)){v.currentTime=at;v.playbackRate=Number($('speed').value);}playing=true;try{await Promise.all(Object.values(videos).map(v=>v.play()));audioNow();$('play').textContent='▶ 播放中';}catch(e){pause();throw e;}}
function seek(t){t=Math.max(0,Math.min(C.duration,t));for(const v of Object.values(videos))v.currentTime=t;$('seek').value=t;if(playing)audioNow();}
function setFocus(value){focus=value;document.querySelectorAll('.videoCard').forEach(el=>el.classList.toggle('focus',el.dataset.pane===value));document.querySelectorAll('[data-focus]').forEach(el=>el.classList.toggle('active',el.dataset.focus===value));if(playing)audioNow();else $('audioStatus').textContent='当前选中：'+value.toUpperCase();}
function intervalText(list){return list.length?list.map(x=>`${x[0].toFixed(2)}–${x[1].toFixed(2)}s`).join('，'):'无活动';}
function strip(model){const v=model.metrics,a=model.audio;return `PSNR ${fmt(v.psnr,2)} · SSIM ${fmt(v.ssim)} · LPIPS ${fmt(v.lpips)}\n表情 ${pct(v.emo_acc)} · ΔResp ${fmt(v.delta_resp)} · 活动 ${a.generated.acoustic.reduce((s,x)=>s+x[1]-x[0],0).toFixed(2)}s`+(Object.keys(model.errors).length?' · 存在指标缺覆盖':'');}
function metricRow(record){return [fmt(record.psnr,2),fmt(record.ssim),fmt(record.fid,2),fmt(record.fvd,2),fmt(record.lpips),pct(record.emo_acc),fmt(record.emo_div_generated,2),fmt(record.delta_resp),record.generated_active_clips+'/200',pct(record.missed_response_rate),pct(record.false_sound_clip_rate),pct(record.strict_audio_f1)];}
function buildOverview(){
 const columns=['模型／步数','PSNR ↑','SSIM ↑','FID ↓','FVD ↓','LPIPS ↓','Emo-Acc ↑','Emo-Div','ΔResp ↓','生成有声片段','GT有声却全静音 ↓','GT无活动却发声 ↓','活动严格F1 ↑'];
 const keys=['psnr','ssim','fid','fvd','lpips','emo_acc',null,'delta_resp',null,'missed_response_rate','false_sound_clip_rate','strict_audio_f1'];
 const up=new Set(['psnr','ssim','emo_acc','strict_audio_f1']);
 const best=keys.map(k=>k?(up.has(k)?Math.max:Math.min)(...D.summary.map(r=>r[k])):null);
 $('summaryTable').innerHTML='<thead><tr>'+columns.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+D.summary.map(r=>'<tr><td>'+r.label+'</td>'+metricRow(r).map((x,i)=>'<td class="'+(keys[i]&&r[keys[i]]===best[i]?'best':'')+'">'+x+'</td>').join('')+'</tr>').join('')+'</tbody>';
 const rows=[];for(const [name,p] of Object.entries(D.paired)){for(const [metric,s] of Object.entries(p.metrics)){const scale=metric==='emo_acc'?100:1;rows.push(`<tr><td>${D.models.find(x=>x.key===p.left).label} → ${D.models.find(x=>x.key===p.right).label}</td><td>${metric}${scale===100?' (百分点)':''}</td><td>${fmt(s.difference_right_minus_left*scale,4)}</td><td>[${fmt(s.ci95[0]*scale,4)}, ${fmt(s.ci95[1]*scale,4)}]</td><td>${s.pairs}</td></tr>`);}}
 $('pairedTable').innerHTML='<tr><th>右侧−左侧</th><th>指标</th><th>配对差值</th><th>源group bootstrap95%区间</th><th>共同有效条数</th></tr>'+rows.join('');
}
function buildCaseTable(){
 $('allMetrics').innerHTML='<tr><th>模型</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th><th>Emo-Acc ↑</th><th>ΔResp ↓</th><th>有声时长</th><th>切换</th></tr>'+D.models.map(meta=>{const r=C.models[meta.key],m=r.metrics;return `<tr><td>${meta.label}</td><td>${fmt(m.psnr,2)}</td><td>${fmt(m.ssim)}</td><td>${fmt(m.lpips)}</td><td>${pct(m.emo_acc)}</td><td>${fmt(m.delta_resp)}</td><td>${r.audio.generated.acoustic.reduce((s,x)=>s+x[1]-x[0],0).toFixed(2)}s</td><td><button data-set-a="${meta.key}">设为A</button> <button data-set-b="${meta.key}">设为B</button></td></tr>`;}).join('');
 $('allMetrics').querySelectorAll('[data-set-a],[data-set-b]').forEach(el=>el.onclick=()=>{const pane=el.dataset.setA?'A':'B';$('model'+pane).value=el.dataset.setA||el.dataset.setB;setModels().catch(fail);});
}
function metadata(v){return new Promise((resolve,reject)=>{if(v.readyState>=1)return resolve();const timer=setTimeout(()=>reject(Error('视频元信息加载超时')),20000);v.addEventListener('loadedmetadata',()=>{clearTimeout(timer);resolve();},{once:true});v.addEventListener('error',()=>{clearTimeout(timer);reject(Error('视频加载失败 '+v.currentSrc));},{once:true});});}
async function setModels(){
 const version=++loadVersion;pause();changing=true;$('play').disabled=true;$('error').textContent='';$('loading').textContent='正在加载视频和校验真实PCM音轨…';
 const a=C.models[$('modelA').value],b=C.models[$('modelB').value];
 for(const [key,url] of Object.entries({gt:C.gt_video,a:a.video,b:b.video})){videos[key].src=url;videos[key].muted=true;videos[key].load();}
 const result=await Promise.all([getPCM(C.gt_pcm),getPCM(C.speaker_pcm),getPCM(a.pcm),getPCM(b.pcm),...Object.values(videos).map(metadata)]);
 if(version!==loadVersion)return;buffers={gt:result[0],speaker:result[1],a:result[2],b:result[3]};
 $('metricsA').textContent=strip(a);$('metricsB').textContent=strip(b);
 $('metricsGT').textContent=`时长 ${C.duration.toFixed(2)}s · 原始脸短边p10 ${C.face_p10.toFixed(0)}px\n净化GT${C.gt_pcm.exact_zero?'严格全零':'含非零波形'} · 声学活动 ${intervalText(C.gt_audio_intervals.acoustic)}`;
 for(const [id,url] of Object.entries({downloadGT:C.gt_video,downloadGTwav:C.gt_wav,downloadSpeaker:C.speaker_wav,downloadA:a.video,downloadAwav:a.wav,downloadB:b.video,downloadBwav:b.wav}))$(id).href=url;
 $('intervals').textContent='声学活动区间（同一pair秒数）：GT '+intervalText(C.gt_audio_intervals.acoustic)+' ｜ A '+intervalText(a.audio.generated.acoustic)+' ｜ B '+intervalText(b.audio.generated.acoustic);
 changing=false;for(const id of ['play','pause','reset','seek','samples','modelA','modelB','prev','next'])$(id).disabled=false;
 $('seek').max=C.duration;$('seek').value=0;seek(0);$('loading').textContent='已加载。音频与正式评测的float32波形SHA一致。';
 const url=new URL(location);url.searchParams.set('sample',C.id);url.searchParams.set('a',$('modelA').value);url.searchParams.set('b',$('modelB').value);history.replaceState(null,'',url);loadNote();
}
async function setCase(index,initial=false){
 pause();C=D.cases[index];$('samples').value=C.id;$('modelA').value=C.a;$('modelB').value=C.b;
 if(initial){const q=new URL(location).searchParams;for(const k of ['a','b'])if(D.models.some(x=>x.key===q.get(k)))$('model'+k.toUpperCase()).value=q.get(k);}
 $('caseCount').textContent=(index+1)+' / '+D.cases.length;$('sampleInfo').textContent=`${C.id} · 源时间 ${C.source_start.toFixed(2)}–${C.source_end.toFixed(2)}s · ${C.source_group}`;
 $('caption').textContent=C.gt_caption;$('rule').textContent='选例依据：'+C.selection_rule+'。本条尚未人工质量验收，标题描述的是自动统计差异。';$('firstFrame').src=C.first_frame;
 $('reviewFocus').textContent='复核重点：'+(C.review_focus||C.selection_rule);
 $('provenance').textContent='实际修正裁框 '+JSON.stringify(C.media_audit.final_crop)+'；首帧与推理缓存逐像素匹配。GT RGB SHA '+C.media_audit.gt_rgb_sha256;
 $('basePlan').textContent=C.base_plan.description;$('sftPlan').textContent=C.sft_plan.description;
 $('fullPrompt').textContent='原生Gemma prompt\n'+C.base_plan.prompt+'\n\nSFT Gemma prompt\n'+C.sft_plan.prompt+'\n\nQwen system\n'+C.base_plan.system+'\n\nQwen user\n'+C.base_plan.user;
 buildCaseTable();setFocus('gt');await setModels();
}
function reviews(){try{return JSON.parse(localStorage.getItem(storage)||'{}');}catch(e){return {};}}
function reviewKey(){return [C.id,$('modelA').value,$('modelB').value].join('|');}
function loadNote(){const r=reviews()[reviewKey()]||{};$('decision').value=r.decision||'';$('note').value=r.note||'';}
function saveNote(){if(!C)return;const all=reviews();all[reviewKey()]={sample_id:C.id,a:$('modelA').value,b:$('modelB').value,decision:$('decision').value,note:$('note').value,updated_at:new Date().toISOString(),human_ground_truth:false};localStorage.setItem(storage,JSON.stringify(all));}
$('play').onclick=()=>play().catch(fail);$('pause').onclick=pause;$('reset').onclick=()=>{pause();seek(0);};$('seek').oninput=()=>seek(Number($('seek').value));$('speed').onchange=()=>{for(const v of Object.values(videos))v.playbackRate=Number($('speed').value);if(playing)audioNow();};
document.querySelectorAll('[data-mode]').forEach(el=>el.onclick=()=>{mode=el.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===el));if(playing)audioNow();});
document.querySelectorAll('[data-focus]').forEach(el=>el.onclick=()=>setFocus(el.dataset.focus));for(const [key,v] of Object.entries(videos))v.onclick=()=>setFocus(key);
$('samples').onchange=()=>setCase(D.cases.findIndex(x=>x.id===$('samples').value)).catch(fail);
$('prev').onclick=()=>setCase((D.cases.indexOf(C)-1+D.cases.length)%D.cases.length).catch(fail);$('next').onclick=()=>setCase((D.cases.indexOf(C)+1)%D.cases.length).catch(fail);
$('modelA').onchange=()=>setModels().catch(fail);$('modelB').onchange=()=>setModels().catch(fail);
const presets={loss:['av_event_6000','base_feature_6000'],combined:['base_feature_6000','base_event_feature_7000'],planner:['av_event_6000','sft_on_base_event_6000'],hidden:['sft_on_base_event_6000','sft_event_6000'],steps:['av_event_6000','av_event_8927']};
document.querySelectorAll('[data-preset]').forEach(el=>el.onclick=()=>{if(!D||changing)return;[$('modelA').value,$('modelB').value]=presets[el.dataset.preset];setModels().catch(fail);});
$('decision').onchange=saveNote;$('note').oninput=saveNote;$('exportReview').onclick=()=>{const b=new Blob([JSON.stringify({schema:'seamless200-tenway-human-review-v1',exported_at:new Date().toISOString(),human_ground_truth:false,records:Object.values(reviews())},null,2)],{type:'application/json'});const url=URL.createObjectURL(b),a=document.createElement('a');a.href=url;a.download='seamless200-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
setInterval(()=>{if(!C)return;const t=videos.gt.currentTime;$('seek').value=t;$('time').textContent=t.toFixed(2)+' / '+C.duration.toFixed(2)+'s';if(playing){if(t>=C.duration-.03){pause();return;}for(const key of ['a','b'])if(Math.abs(videos[key].currentTime-t)>.09)videos[key].currentTime=t;}},120);
window.reviewDebug=()=>({sample:C?.id,focus,mode,playing,context:ctx?.state,sources:sources.length,times:Object.fromEntries(Object.entries(videos).map(([k,v])=>[k,v.currentTime])),loaded:!changing});
async function init(){const r=await fetch('data.json');if(!r.ok)throw Error('结果数据HTTP '+r.status);D=await r.json();window.reviewData=D;buildOverview();for(const c of D.cases){const o=new Option(String(c.index).padStart(2,'0')+' · '+c.category+' · '+c.title,c.id);$('samples').add(o);}for(const meta of D.models){$('modelA').add(new Option(meta.label,meta.key));$('modelB').add(new Option(meta.label,meta.key));}const requested=new URL(location).searchParams.get('sample');const index=Math.max(0,D.cases.findIndex(x=>x.id===requested));await setCase(index,true);}
init().catch(fail);
