'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>Number(n).toFixed(3);
const STORE='listen-native-fps-light-review-20261029-v1';
const issueLabels=['动作遗漏','不存在的动作','一次动作切成多次','多次动作合成一次','边界偏差明显','双方共同漏标','无明显问题'];
let data,sample,chosen=null,review={},storageOK=true,filtered=[];
try{review=JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){storageOK=false}
const armLabel=a=>a==='video8'?'原生视频8 fps':'原生视频16 fps';
const slotLabel=i=>i===0?'甲':'乙';
function qualified(){return Object.values(review).filter(r=>r.decision!==''&&r.decision!=null)}
function updateProgress(){const done=qualified().filter(r=>data.queue.includes(r.sample_id)).length;$('progress').textContent=`推荐项已复核 ${done} / ${data.queue.length}`}
function candidates(){
 const q=$('queue').value,search=$('search').value.toLowerCase();
 const ids=q==='review'?data.queue:data.samples.filter(s=>q==='all'||s.task.group===q).map(s=>s.id);
 return ids.map(id=>data.samples.find(s=>s.id===id)).filter(s=>s.id.toLowerCase().includes(search));
}
function filterSamples(){
 filtered=candidates();$('sampleSelect').innerHTML=filtered.map(s=>`<option value="${esc(s.id)}">${review[s.task.id]?.decision?'✓ ':''}${esc(s.short)} · ${esc(s.task.title)} · ${esc(s.id.split('_local_')[0])}</option>`).join('');
 $('sampleSelect').value=sample?.id||'';updatePosition();
}
function updatePosition(){const i=filtered.findIndex(s=>s.id===sample?.id);$('position').textContent=i<0?'不在当前筛选':`${i+1} / ${filtered.length}`;$('prev').disabled=i<=0;$('next').disabled=i<0||i>=filtered.length-1}
function switchBy(delta){const i=filtered.findIndex(s=>s.id===sample?.id);if(i+delta>=0&&i+delta<filtered.length)selectSample(filtered[i+delta].id)}
function selectSample(id){
 sample=data.samples.find(s=>s.id===id)||filtered[0]||data.samples[0];chosen=null;$('video').pause();
 window.reviewAudioLoad(sample);$('video').src=sample.video;$('video').playbackRate=+$('speed').value;
 $('sid').textContent=sample.id;$('sampleSelect').value=sample.id;updatePosition();
 $('mediaMeta').textContent=JSON.stringify({source_start_s:sample.source_start_s,duration_s:sample.duration,video_transform:sample.crop,source_sha256:sample.source_sha256},null,2);
 $('taskType').textContent=sample.task.group==='key'?'关键分歧 · 定向选择':sample.task.group==='random'?'固定随机抽查 · 双方文字相近':'全部样本 · 自由复核';
 $('taskTitle').textContent=sample.task.title;$('question').textContent=sample.task.question;
 $('choices').innerHTML=sample.task.options.map((s,i)=>`<label><input type="radio" name="decision" value="${i}"> ${esc(s)}</label>`).join('');
 $('choices').querySelectorAll('input').forEach(e=>e.onchange=saveReview);
 $('candidateBox').open=false;renderCandidate();
 $('view').innerHTML=sample.series.video8.map(r=>`<option value="${esc(r.view)}">${r.view==='full'?'整段':esc(r.view)} · ${fmt(r.start)}–${fmt(r.end)}秒</option>`).join('');$('view').value='full';
 renderColumns();loadReview();$('playState').textContent=`待核区间 ${fmt(sample.task.range[0])}–${fmt(sample.task.range[1])} 秒`;
 const url=new URL(location.href);url.searchParams.set('sample',sample.id);history.replaceState(null,'',url);
}
function renderCandidate(){
 let text=sample.task.candidate;
 for(const a of ['video8','video16']){
  const name=$('reveal').checked?armLabel(a):slotLabel(sample.assignment.indexOf(a));text=text.replaceAll(a,name);
 }
 $('candidate').textContent=text;$('evidence').textContent=sample.task.evidence?'依据：已抽取视频帧进行观察；仍需你确认动作次数与边界。':'抽样规则：文本相似度≥0.90，起止差≤0.2秒，固定种子按片段抽取；不等于语义真值。';
}
function renderColumns(){
 const view=$('view').value;
 $('columns').innerHTML=sample.assignment.map((arm,slot)=>{
  const r=sample.series[arm].find(r=>r.view===view),valid=r.status==='valid';const es=valid?r.parsed.events:[];
  const title=slotLabel(slot)+($('reveal').checked?' · '+armLabel(arm):' · 配置已隐藏');
  return `<article class="panel card" data-slot="${slotLabel(slot)}"><div class="cardhead"><h3>${esc(title)}</h3><p class="meta">${valid?es.length+' 条原始候选':'结构未通过'} · 不是动作真值</p></div><div class="events">${!valid?'<div class="bad">本请求无效／可能截断，查看下方完整原始响应。</div>':es.map((e,i)=>{
   const a=e.start+r.start,b=e.end+r.start;const refs=sample.task.matched_indices;const focus=view==='full'&&refs&&i===refs[arm==='video8'?0:1];
   return `<div class="event${focus?' focus':''}" data-start="${a}" data-end="${b}"><button class="seek" data-a="${a}" data-b="${b}">${fmt(a)}–${fmt(b)} 秒 ↻</button> <span class="meta">#${i+1}${focus?' · 本次随机抽查项':''}</span><p class="behavior">${esc(e.behavior)}</p></div>`;
  }).join('')}</div><div class="raw"><details><summary>完整原始响应／prompt</summary><p class="meta">原始JSON使用窗口内时间，下方未改写。</p><pre>${esc(r.raw)}</pre><pre>${esc(r.prompt)}</pre></details></div></article>`;
 }).join('');
 document.querySelectorAll('.seek').forEach(b=>b.onclick=()=>playInterval(+b.dataset.a,+b.dataset.b,true));
}
function playInterval(a,b,context=false){chosen=[Math.max(0,a-(context ? .3 : 0)),Math.min(sample.render_duration||sample.duration,b+(context ? .3 : 0))];$('video').currentTime=chosen[0];$('video').play().catch(e=>$('playState').textContent='请点播放键：'+e.message);$('playState').textContent=`播放 ${fmt(a)}–${fmt(b)} 秒`}
function loadReview(){
 const r=review[sample.task.id]||{};
 document.querySelectorAll('[name=decision]').forEach(e=>e.checked=e.value===r.decision);
 $('preference').value=r.preference||'';$('note').value=r.note||'';
 document.querySelectorAll('#issues input').forEach(e=>e.checked=(r.issues||[]).includes(e.value));
 $('saved').textContent=r.updated_at?'已保存本条意见':'';
}
function saveReview(){
 const decision=document.querySelector('[name=decision]:checked')?.value??'';
 review[sample.task.id]={case_id:sample.task.id,sample_id:sample.id,group:sample.task.group,question:sample.task.question,
  decision,answer:decision===''?null:sample.task.options[Number(decision)],preference:$('preference').value,
  preferred_arm:['A','B'].includes($('preference').value)?sample.assignment[$('preference').value==='A'?0:1]:null,
  blind_assignment:{A:sample.assignment[0],B:sample.assignment[1]},names_revealed:$('reveal').checked||!!review[sample.task.id]?.names_revealed,
  candidate_judgment_opened:!!sample._candidateOpened||!!review[sample.task.id]?.candidate_judgment_opened,view:$('view').value,
  issues:[...document.querySelectorAll('#issues input:checked')].map(e=>e.value),note:$('note').value,
  source_sha256:sample.source_sha256,human_ground_truth:false,training_admitted:false,updated_at:new Date().toISOString()};
 try{localStorage.setItem(STORE,JSON.stringify(review));$('saved').textContent=decision===''?'草稿已保存；尚未选择答案':'本条已保存'}catch(e){storageOK=false;$('saved').textContent='浏览器存储失败，请立即导出JSON'}
 updateProgress();
}
async function refreshSpeed(){
 try{
  const r=await fetch('speed/status.json?t='+Date.now(),{cache:'no-store'});
  if(!r.ok){$('speedState').textContent='尚未开始或状态文件未生成';return}
  const s=await r.json();$('speedState').textContent=s.state==='complete'?'测速完成（共享GPU条件）':['failed','interrupted'].includes(s.state)?'测速中断：'+(s.error||s.reason):s.state==='running'?`正在测速：${s.completed_batches}/${s.total_batches} 批`:'正在准备输入／加载模型';
  if(s.state==='complete'){
   $('speedResults').innerHTML='<div class="scroll"><table><thead><tr><th>配置</th><th>请求数</th><th>有效结构</th><th>请求/分钟</th><th>有效请求/分钟</th><th>总推理秒数</th></tr></thead><tbody>'+['video8','video16'].map(a=>{const x=s.totals[a];return `<tr><td>${armLabel(a)}</td><td>${x.requests}</td><td>${x.valid_requests}</td><td>${x.requests_per_minute.toFixed(2)}</td><td>${x.valid_requests_per_minute.toFixed(2)}</td><td>${x.seconds.toFixed(2)}</td></tr>`}).join('')+'</tbody></table></div>'+`<p>16／8 fps推理总耗时比：${s.ratio_16_to_8_wall_time.toFixed(3)}。范围为${s.plan.samples.length}条分层随机样本的${s.plan.views==='full'?'整段':'整段与部分短窗'}、每组重复两遍；batch=${s.plan.batch_size}，不包含模型加载和源视频解码。</p><p class="meta">共享训练负载可能干扰结果，不推算独占GPU或全量完成时间。有效结构也不等于语义正确。</p>`;
  }
 }catch(e){$('speedState').textContent='测速状态读取失败：'+e.message}
}
$('video').addEventListener('timeupdate',()=>{const t=$('video').currentTime;if(chosen&&t>=chosen[1]){if($('loop').checked)$('video').currentTime=chosen[0];else{$('video').pause();chosen=null}}document.querySelectorAll('.event').forEach(e=>e.classList.toggle('now',t>=+e.dataset.start&&t<+e.dataset.end))});
$('fullPlay').onclick=()=>{chosen=null;$('video').currentTime=0;$('video').play().catch(()=>{});$('playState').textContent='播放整段：检查双方是否共同漏标'};
$('focusPlay').onclick=()=>playInterval(...sample.task.range);
$('speed').onchange=()=>$('video').playbackRate=+$('speed').value;
$('view').onchange=renderColumns;$('reveal').onchange=()=>{renderColumns();renderCandidate()};
$('queue').onchange=()=>{filterSamples();if(filtered.length)selectSample(filtered[0].id)};
$('search').oninput=filterSamples;$('sampleSelect').onchange=e=>selectSample(e.target.value);
$('prev').onclick=()=>switchBy(-1);$('next').onclick=()=>switchBy(1);
$('save').onclick=saveReview;$('saveNext').onclick=()=>{saveReview();if(!document.querySelector('[name=decision]:checked')){$('saved').textContent='请选择一个答案；不确定时可选“无法判断”。';return}switchBy(1)};
$('preference').onchange=saveReview;$('note').oninput=saveReview;
$('candidateBox').addEventListener('toggle',()=>{if(sample&&$('candidateBox').open)sample._candidateOpened=true});
$('refreshSpeed').onclick=refreshSpeed;
$('export').onclick=()=>{const payload={schema:'listen-native-fps-human-review-v1',run_id:data.run_id,human_ground_truth:false,training_admitted:false,exported_at:new Date().toISOString(),sampling:data.sampling,records:Object.values(review)};const u=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='native_fps_review_20261029.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
fetch('data.json').then(r=>{if(!r.ok)throw Error('data.json HTTP '+r.status);return r.json()}).then(d=>{
 data=d;window.nativeFPSReview=d;window.nativeFPSState=()=>({sample:sample.id,chosen,view:$('view').value,review});
 $('issues').innerHTML=issueLabels.map(s=>`<label><input type="checkbox" value="${esc(s)}">${esc(s)}</label>`).join('');document.querySelectorAll('#issues input').forEach(e=>e.onchange=saveReview);
 const requested=new URLSearchParams(location.search).get('sample');if(requested&&!d.queue.includes(requested))$('queue').value='all';
 filterSamples();selectSample(requested||d.queue[0]);updateProgress();$('loading').hidden=true;$('app').hidden=false;
 if(!storageOK)$('saved').textContent='浏览器存储不可用，请导出JSON保存';refreshSpeed();
}).catch(e=>{$('loading').textContent='加载失败：'+e.message+'。请确认Cursor转发的是当前机器的8876端口。';$('loading').className='bad'});
