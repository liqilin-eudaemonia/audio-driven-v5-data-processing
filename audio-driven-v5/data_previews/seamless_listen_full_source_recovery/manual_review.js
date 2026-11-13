/* Human review only. Browser feedback never changes production admission. */
(()=>{
const key='listen-source-false-rejection-review-v1';
let notes={};try{notes=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){notes={}}
let all=[],visible=[],selected=null,player=null,manualAudio=null,gainNodes=null;
const label={quality:'音质／阻断疑点',speech:'未提取，但存在转写',other:'其他未提取区间',tiny:'极短间隙'};
const priority={quality:0,speech:1,other:2,tiny:3};
const decisionLabels={pending:'待复核',keep:'误杀，可保留',reject:'正确排除',partial:'部分可保留',uncertain:'听不清／待定'};
const q=s=>document.querySelector('#manualReview '+s);
function noteKey(r){return r.pair+':'+r.id}
function sourceTime(){return selected.context_start+(player.currentTime||0)}
function route(){
  if(!player)return;
  if(!manualAudio){manualAudio=new AudioContext();const src=manualAudio.createMediaElementSource(player),split=manualAudio.createChannelSplitter(2);src.connect(split);gainNodes=[manualAudio.createGain(),manualAudio.createGain()];gainNodes.forEach((g,i)=>{split.connect(g,i);g.connect(manualAudio.destination)})}
  manualAudio.resume();const mode=q('#manualChannel').value;
  gainNodes.forEach((g,i)=>g.gain.setValueAtTime(mode==='both'?.5:mode==='a'?(i===0?1:0):(i===1?1:0),manualAudio.currentTime));
}
function update(){
  if(!selected||!player)return;
  const t=sourceTime(),focus=selected.start<=t&&t<selected.end;
  q('#manualClock').textContent=`源时间 ${fmt(t)} · ${focus?'正在播放需复核区间':'前后上下文（不代表保留候选）'}`;
  const c=q('canvas'),g=c.getContext('2d'),w=c.width,span=selected.context_seconds;
  g.fillStyle='#152239';g.fillRect(0,0,w,66);
  g.fillStyle='#e1ad55';g.fillRect((selected.start-selected.context_start)/span*w,10,Math.max(2,selected.focus_seconds/span*w),25);
  g.strokeStyle='white';g.beginPath();g.moveTo((t-selected.context_start)/span*w,0);g.lineTo((t-selected.context_start)/span*w,42);g.stroke();
  g.font='15px sans-serif';g.fillStyle='#dae5f5';g.fillText(fmt(selected.context_start),5,60);g.fillText(fmt(selected.context_end),w-115,60);
  const role=selected.regions.find(r=>r.start<=t&&t<r.end);
  q('#manualRole').textContent=role?`此时角色：${role.speaker_index?'B 说话，A 倾听':'A 说话，B 倾听'}`:'此时没有确定的主说话角色；请结合上下文判断。';
}
function save(){
  if(!selected)return;
  const start=Number(q('#manualKeepStart').value),end=Number(q('#manualKeepEnd').value),decision=q('#manualDecision').value;
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<selected.context_start||end>selected.context_end+1e-6||end<=start){q('#manualSaved').textContent='请填写上下文范围内的有效起止时间。';return}
  notes[noteKey(selected)]={id:selected.id,pair:selected.pair,focus_start:selected.start,focus_end:selected.end,
    context_start:selected.context_start,context_end:selected.context_end,decision,
    suggested_start:start,suggested_end:end,note:q('#manualNote').value.trim(),utc:new Date().toISOString(),
    production_result_sha256:selected.production_result_sha256};
  try{localStorage.setItem(key,JSON.stringify(notes));q('#manualSaved').textContent='已保存到当前浏览器；请导出 JSON 交回复核结论。'}catch(e){q('#manualSaved').textContent='浏览器保存失败，请立即导出 JSON。'}
  renderList();q('#manualProgress').textContent=`已记录 ${all.filter(r=>notes[noteKey(r)]?.decision&&notes[noteKey(r)].decision!=='pending').length}/${all.length} 段结论。`;
}
function exportNotes(){
  const rows=Object.values(notes).filter(r=>r.decision!=='pending'||r.note);
  download(new Blob([JSON.stringify({revision:'human-false-rejection-review-v1',created_utc:new Date().toISOString(),
    decisions:rows,production_modified:false},null,2)],{type:'application/json'}),'疑似误杀人工复核.json');
}
function evidenceHTML(r){
  const timeline=r.timeline.map(t=>`<tr><td>${fmt(t.start)}–${fmt(t.end)}</td><td>${esc(labels[t.status]||t.status)}</td><td>${esc(reasonText(t.reasons))}</td></tr>`).join('');
  const parents=r.evidence.map(e=>`<div class="note"><strong>${esc(e.pool==='audio_review'?'模型疑点，待人工确认':e.pool==='audio_reject'?'规则判定排除，仍可人工复核':e.pool)}</strong> · 原候选 ${fmt(e.parent_start)}–${fmt(e.parent_end)}<p>${esc(reasonText(e.reasons))}</p>${e.model_windows.map(w=>`<p>源 ${fmt(w.start)}–${fmt(w.end)} · ${esc(w.participant)}（当时 ${esc(w.role)}）：${esc(w.noise_label)} 分数 ${w.noise_score.toFixed(3)}，${esc(w.human_label)} 分数 ${w.human_score.toFixed(3)}，RMS ${w.rms_dbfs.toFixed(1)} dBFS。</p>`).join('')}${e.reference_windows.map(w=>`<p>可疑双轨匹配 ${fmt(w.start)}–${fmt(w.end)}：相关性 ${w.correlation.toFixed(3)}，延迟 ${w.delay_ms.toFixed(1)} ms；不等于已确认串音。</p>`).join('')}</div>`).join('');
  return `<table><thead><tr><th>需复核源区间</th><th>当前状态</th><th>规则记录</th></tr></thead><tbody>${timeline}</tbody></table>${parents||'<p>此段没有音质排除证据；只是没有形成候选。请检查是否有可保留的讲话或倾听反应。</p>'}`;
}
function renderList(){q('#manualList').innerHTML=visible.map(r=>`<button class="${selected?.id===r.id?'selected':''}" data-review-id="${r.id}">${esc(r.id)} · ${fmt(r.start)}–${fmt(r.end)}<small>${label[r.category]} · ${r.focus_seconds.toFixed(3)} 秒 · ${decisionLabels[notes[noteKey(r)]?.decision||'pending']}</small></button>`).join('');q('#manualList').querySelectorAll('button').forEach(b=>b.onclick=()=>select(b.dataset.reviewId))}
function select(id){
  const r=visible.find(x=>x.id===id);if(!r)return;selected=r;window.manualReviewCurrent=r;
  if(player)player.pause();if(manualAudio){manualAudio.close();manualAudio=null;gainNodes=null}
  const n=notes[noteKey(r)]||{};
  q('#manualPlayer').innerHTML=`<strong>${esc(r.id)} · ${label[r.category]}</strong><p>需复核 ${fmt(r.start)}–${fmt(r.end)}；视频包含前后最多 3 秒上下文。黄色边框／时间条标出需复核区间。</p><video controls playsinline preload="metadata" poster="${esc(r.poster)}" src="${esc(r.media)}"></video><canvas width="1000" height="66" class="manual-strip"></canvas><p id="manualClock" class="manual-mark"></p><p id="manualRole"></p><div class="controls"><button id="manualReplay">从上下文开头播放</button><button id="manualFocus">播放需复核位置</button><select id="manualChannel" aria-label="复核音轨"><option value="both">双人音轨</option><option value="a">只听 A（左画面）</option><option value="b">只听 B（右画面）</option></select><button id="manualPrev">上一段</button><button id="manualNext">下一段</button></div><div class="manual-feedback"><strong>你的复核结论</strong><div class="controls"><select id="manualDecision" aria-label="人工复核结论">${Object.entries(decisionLabels).map(([k,v])=>`<option value="${k}" ${k===(n.decision||'pending')?'selected':''}>${v}</option>`).join('')}</select><label>建议起点（源秒）<input id="manualKeepStart" type="number" step=".001" value="${n.suggested_start??r.start}" style="width:105px"></label><label>建议终点<input id="manualKeepEnd" type="number" step=".001" value="${n.suggested_end??r.end}" style="width:105px"></label></div><textarea class="manual-note" id="manualNote" placeholder="例如：不是杂音，是正常回应；只有前半段可保留；这里确有串音。">${esc(n.note||'')}</textarea><div class="controls"><button id="manualSave">保存本段结论</button><button id="manualExport">导出两条来源的全部复核结论</button></div><p id="manualSaved" class="muted">保存于当前浏览器。导出 JSON 后可用于后续修订，当前生产数据不会自动改变。</p></div><details open><summary>为什么未纳入：具体规则与模型证据</summary><p>模型分数不是人工确认，也不是“存在噪声的概率”；上下文可能包含已经保留的片段。</p><div class="manual-evidence">${evidenceHTML(r)}</div></details><details><summary>原始转写与下载</summary><div class="manual-transcript">${r.transcripts.map((ss,i)=>`<p><strong>${'AB'[i]}</strong><br>${ss.map(x=>`${fmt(x.start)}–${fmt(x.end)} ${esc(x.text)}`).join('<br>')}</p>`).join('')}</div><p><a href="${esc(r.media)}" download>下载本段视频</a> · <a href="${esc(r.audio_urls.A)}" download>A 独立 WAV</a> · <a href="${esc(r.audio_urls.B)}" download>B 独立 WAV</a> · <a href="${esc(r.evidence_url)}" download>完整证据 JSON</a></p><p>试听沿用整轨响度归一化；没有新增降噪、静音或事件抑制。</p></details>`;
  player=q('video');player.addEventListener('play',route);player.addEventListener('timeupdate',update);player.addEventListener('seeked',update);player.addEventListener('loadedmetadata',update);
  q('#manualChannel').onchange=route;
  q('#manualReplay').onclick=()=>{player.currentTime=0;player.play().catch(showError)};
  q('#manualFocus').onclick=()=>{player.currentTime=Math.max(0,r.start-r.context_start-.3);player.play().catch(showError)};
  q('canvas').onclick=e=>{const rect=e.target.getBoundingClientRect();player.currentTime=Math.max(0,Math.min(r.context_seconds,(e.clientX-rect.left)/rect.width*r.context_seconds))};
  q('#manualSave').onclick=save;q('#manualExport').onclick=exportNotes;
  q('#manualPrev').onclick=()=>select(visible[Math.max(0,visible.indexOf(r)-1)].id);
  q('#manualNext').onclick=()=>select(visible[Math.min(visible.length-1,visible.indexOf(r)+1)].id);
  renderList();update();
}
window.renderManualReview=()=>{
  if(player)player.pause();if(manualAudio){manualAudio.close();manualAudio=null}player=null;selected=null;
  all=data.manual_review||[];const el=document.querySelector('#manualReview');el.hidden=!all.length;if(!all.length)return;
  el.querySelector('summary').textContent=`历史未覆盖区间复核（修复前） · 本条来源 ${all.length} 段未纳入区间`;
  const done=all.filter(r=>notes[noteKey(r)]?.decision&&notes[noteKey(r)].decision!=='pending').length;
  q('#manualBody').innerHTML=`<p>这是修复前的未覆盖区间，不是修复后完整 pair；最新候选请看上方“修复后完整 pair”。历史未纳入区间已全部单独截取，按音质／阻断疑点、有转写未提取、其余区间排序；不预判它们都是误杀。<span id="manualProgress">已记录 ${done}/${all.length} 段结论。</span>切换上方源视频可查看另一条来源。</p><div class="controls"><select id="manualFilter" aria-label="复核类别"><option value="all">全部 ${all.length} 段</option>${Object.entries(label).map(([k,v])=>`<option value="${k}">${v}（${all.filter(r=>r.category===k).length}）</option>`).join('')}</select><a href="?pair=naturalistic_V02_S3579_I00000175&review=1#manualReview">190 秒来源</a><a href="?pair=naturalistic_V03_S1909_I00000205&review=1#manualReview">354 秒来源</a></div><div class="manual-layout"><aside id="manualList" class="manual-list"></aside><section id="manualPlayer" class="manual-player"></section></div>`;
  function filter(){const mode=q('#manualFilter').value;visible=all.filter(r=>mode==='all'||r.category===mode).sort((a,b)=>priority[a.category]-priority[b.category]||a.start-b.start);if(visible.length)select(visible[0].id);else{if(player)player.pause();if(manualAudio){manualAudio.close();manualAudio=null;gainNodes=null}player=null;window.manualReviewCurrent=null;q('#manualList').innerHTML='此类别没有片段';q('#manualPlayer').innerHTML='';selected=null}}
  q('#manualFilter').onchange=filter;filter();
  if(new URLSearchParams(location.search).get('review')==='1'){el.open=true;requestAnimationFrame(()=>el.scrollIntoView({block:'start'}))}
};
if(window.reviewData)window.renderManualReview();
})();
