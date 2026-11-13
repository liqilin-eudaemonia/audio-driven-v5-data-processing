/* Repaired whole-pair review. Feedback is separate from historical gap notes. */
(()=>{
const storageKey='listen-repaired-whole-pairs-review-v3';
const poolNames={keep_candidate:'规则通过',audio_review:'待音质复核',audio_reject:'规则排除',human_hold:'人工暂缓',task_hold:'任务条件暂缓',source_hold:'来源暂缓'};
const verdicts={pending:'待人工复核',keep:'已核音质与边界，整段可保留',reject:'正确排除',partial:'部分可保留',uncertain:'听不清／待定'};
const extraReasons={before_long_gap_unconfirmed:'长停顿前句尾未确认',after_long_gap_unconfirmed:'长停顿后句首未确认',inferred_clause_start:'推测分句开头',inferred_clause_end:'推测分句结尾',transcript_end_unconfirmed:'原转写句尾未确认',source_sentence_start_unconfirmed:'原转写句首未确认',recovered_partial_sentence:'按词边界恢复的句子片段',environmental_prediction_unresolved:'环境声模型疑点尚未确认',intrusive_transient_noise:'规则检测到突发杂音',isolated_reference_match_unresolved:'单窗口双轨相似，疑似串音待确认',reference_match:'双轨相似疑点'};
let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
try{const legacy=JSON.parse(localStorage.getItem('listen-repaired-whole-pairs-review-v2')||'{}');for(const r of Object.values(legacy)){const k=(r.revision||'local-listening-boundary-repair-2')+':'+r.pair+':'+r.sample_id;if(!saved[k])saved[k]=r}}catch(e){}
let set=null,visible=[],current=null,player=null,ac=null,gains=null;
const q=s=>document.querySelector('#fixedPairReview '+s);
const nk=r=>set.revision+':'+r.pair+':'+r.sample_id;
const semanticFlags=new Set(['before_long_gap_unconfirmed','after_long_gap_unconfirmed','inferred_clause_start','inferred_clause_end','transcript_end_unconfirmed','source_sentence_start_unconfirmed','recovered_partial_sentence']);
const semanticReview=r=>(r.boundary_flags||[]).some(x=>semanticFlags.has(x));
const why=rs=>(rs||[]).map(r=>extraReasons[r]||reasonText([r])).join('；');
function stop(){if(player)player.pause();if(ac){ac.close();ac=null;gains=null}player=null}
function audio(){
  if(!player)return;
  if(!ac){ac=new AudioContext();const source=ac.createMediaElementSource(player),split=ac.createChannelSplitter(2);source.connect(split);gains=[ac.createGain(),ac.createGain()];gains.forEach((g,i)=>{split.connect(g,i);g.connect(ac.destination)})}
  ac.resume();const mode=q('#fixedChannel').value;
  gains.forEach((g,i)=>g.gain.setValueAtTime(mode==='both'?.5:mode==='a'?(i===0?1:0):(i===1?1:0),ac.currentTime));
}
function draw(){
  if(!current||!player)return;
  const t=current.start+player.currentTime,c=q('#fixedStrip'),g=c.getContext('2d'),w=c.width;
  g.fillStyle='#172a40';g.fillRect(0,0,w,66);
  g.fillStyle=current.pool==='keep_candidate'?'#4b9b79':'#d79b49';
  g.fillRect(0,18,w,12);
  g.fillStyle='#f9c76e';
  current.evidence_windows.forEach(x=>g.fillRect(Math.max(0,(x.start-current.start)/current.duration*w),8,Math.min(w,(x.end-x.start)/current.duration*w),32));
  g.strokeStyle='white';g.beginPath();g.moveTo(player.currentTime/current.duration*w,0);g.lineTo(player.currentTime/current.duration*w,44);g.stroke();
  g.fillStyle='#d9e6f5';g.font='15px sans-serif';g.fillText(fmt(current.start),5,61);g.fillText(fmt(current.end),w-115,61);
  const hits=current.evidence_windows.filter(x=>x.start<=t&&t<x.end);
  q('#fixedClock').textContent=`源时间 ${fmt(t)} · pair 内 ${player.currentTime.toFixed(2)} / ${current.duration.toFixed(3)} 秒${hits.length?' · 当前处于规则疑点窗口':''}`;
}
function progress(){q('#fixedProgress').textContent=`已记录 ${set.rows.filter(r=>saved[nk(r)]?.decision&&saved[nk(r)].decision!=='pending').length} 段人工结论`}
function list(){
  q('#fixedList').innerHTML=visible.map(r=>`<button class="${current?.id===r.id?'selected':''}" data-fixed-id="${r.id}"><b>${r.id} · ${r.duration.toFixed(3)} 秒</b><small>${fmt(r.start)}–${fmt(r.end)} · ${poolNames[r.pool]||r.pool}${r.initial?'':' · 局部恢复子段'}</small><small>${esc(saved[nk(r)]?verdicts[saved[nk(r)].decision]:why(q('#fixedFilter').value==='semantic'?r.boundary_flags.filter(x=>semanticFlags.has(x)):r.reasons))}</small></button>`).join('');
  q('#fixedList').querySelectorAll('button').forEach(b=>b.onclick=()=>select(b.dataset.fixedId));
}
function save(){
  const a=Number(q('#fixedStart').value),b=Number(q('#fixedEnd').value),decision=q('#fixedDecision').value;
  if(!Number.isFinite(a)||!Number.isFinite(b)||a<current.start-1e-6||b>current.end+1e-6||b<=a){q('#fixedSaved').textContent='建议范围必须在本完整 pair 内，且终点大于起点。';return}
  saved[nk(current)]={pair:current.pair,sample_id:current.sample_id,id:current.id,revision:set.revision,
    start:current.start,end:current.end,decision,suggested_start:decision==='keep'?current.start:a,
    suggested_end:decision==='keep'?current.end:b,note:q('#fixedNote').value.trim(),utc:new Date().toISOString(),
    reviewed_whole_pair:true,automatic_pool:current.pool,production_modified:false};
  try{localStorage.setItem(storageKey,JSON.stringify(saved));q('#fixedSaved').textContent='已保存。选择“整段可保留”时，结论覆盖整段播放范围；请导出 JSON。'}catch(e){q('#fixedSaved').textContent='浏览器保存失败，请立即导出 JSON。'}
  list();progress();
}
function exportFeedback(){download(new Blob([JSON.stringify({revision:set.revision,scope:set.review_scope||'two repaired source conversations',
  rows:Object.values(saved).filter(r=>r.decision!=='pending'||r.note),production_modified:false},null,2)],{type:'application/json'}),'修复后完整pair人工复核.json')}
function select(id){
  const r=set.rows.find(x=>x.id===id);if(!r)return;stop();current=r;window.fixedPairCurrent=r;
  const n=saved[nk(r)]||{},si='AB'[r.speaker_index],li='AB'[1-r.speaker_index];
  const tokens=r.text.split(/\s+/),edgeText=tokens.length>20?tokens.slice(0,8).join(' ')+' … '+tokens.slice(-10).join(' '):r.text;
  q('#fixedPlayer').innerHTML=`<strong>${r.id} · ${poolNames[r.pool]||r.pool} · 候选时长 ${r.duration.toFixed(3)} 秒</strong>
    <p class="fixed-range">本次复核对象就是整段视频：${fmt(r.start)}–${fmt(r.end)}，没有额外上下文；整段播放不代表语义完整。${si} 说话，${li} 倾听。</p>
    <video controls playsinline preload="metadata" src="${esc(r.media)}" poster="${esc(r.poster)}"></video>
    <canvas id="fixedStrip" class="manual-strip" width="1000" height="66"></canvas><p id="fixedClock" class="manual-mark"></p>
    <div class="controls"><button id="fixedReplay">播放完整 pair</button><button id="fixedHead">听开头</button><button id="fixedTail">听结尾</button><button id="fixedSourceContext">原视频句首附近</button><button id="fixedSourceTail">原视频句尾附近</button><select id="fixedChannel" aria-label="完整 pair 音轨"><option value="both">双人音轨</option><option value="a">只听 A（左画面）</option><option value="b">只听 B（右画面）</option></select><button id="fixedPrev">上一段</button><button id="fixedNext">下一段</button></div>
    <p class="fixed-edge-text"><strong>片段首尾转写：</strong>${esc(edgeText)}</p>
    <p class="fixed-semantic"><strong>语义边界：</strong>${semanticReview(r)?`尚未确认 · ${esc(why(r.boundary_flags.filter(x=>semanticFlags.has(x))))}。请检查句首、句尾是否完整。`:'未记录语义边界提示；仍可结合原视频核对。'}</p>
    <p><strong>当前规则结论：</strong>${esc(why(r.reasons))}。${r.pool==='keep_candidate'?'通过现有规则，仍可人工检查。':'属于待人工确认的候选，不能仅凭模型分数断言确有污染。'}</p>
    ${r.children.length?`<p class="note">该完整父 pair 未整体通过，但局部恢复已产生：${r.children.map(x=>`<button data-child="${x.id}">${x.id} · ${fmt(x.start)}–${fmt(x.end)} · ${poolNames[x.pool]||x.pool}</button>`).join(' ')}。如确认父 pair 可保留，后续应替换重叠子段，避免重复纳入。</p>`:''}
    ${r.semantic_recovery_hold?`<p class="note"><strong>本次保留整段供人工复核。</strong>恢复提案不能确认句首与句末完整，未用它们替代本段；本段音质结论仍为${poolNames[r.pool]||r.pool}。</p>`:''}
    ${r.parent_id?`<p class="note">这是局部恢复子段，完整父 pair：<button data-child="${r.parent_id}">${r.parent_id}</button>。</p>`:''}
    <details open><summary>导致暂缓／排除的具体位置</summary><p>黄色时间条表示规则定位的疑点。Music／Speech 等是模型独立类别分数，不是音质评分、声音占比或已校准概率。</p>
    <div class="controls">${r.evidence_windows.map(w=>`<button data-evidence-time="${w.start}">${fmt(w.start)}–${fmt(w.end)} · ${esc(why([w.reason]))}</button>`).join('')||'无导致暂缓或排除的定位窗口。'}</div>
    <div class="manual-evidence">${r.model_windows.map(w=>`<p>源 ${fmt(w.start)}–${fmt(w.end)} · ${w.participant}（${w.role==='speaker'?'说话者':'倾听者'}）：${esc(w.noise_label)} 分数 ${w.noise_score.toFixed(3)}，${esc(w.human_label)} 分数 ${w.human_score.toFixed(3)}；RMS ${w.rms_dbfs.toFixed(1)} dBFS。</p>`).join('')}
    ${r.reference_windows.map(w=>`<p>源 ${fmt(w.start)}–${fmt(w.end)}：双轨相关性 ${w.correlation.toFixed(3)}，延迟 ${w.delay_ms.toFixed(1)} ms；相似性尚未由人工确认为串音。</p>`).join('')}</div>
    <p>完整 pair 的最终理由见上方；单个模型窗口告警可能已被放宽规则处理。${r.quality_provenance.reused?'本段边界及归一化音轨与既有结果完全一致，复用原音质证据，并重新应用当前路由。':'本段已对实际新边界重新计算 QC3／QC4。'}</p></details>
    <div class="manual-feedback"><strong>人工复核结论（针对完整 pair）</strong><div class="controls"><select id="fixedDecision" aria-label="完整 pair 复核结论">${Object.entries(verdicts).map(([k,v])=>`<option value="${k}" ${k===(n.decision||'pending')?'selected':''}>${v}</option>`).join('')}</select><label>部分保留起点<input id="fixedStart" type="number" step=".001" value="${n.suggested_start??r.start}" style="width:110px"></label><label>终点<input id="fixedEnd" type="number" step=".001" value="${n.suggested_end??r.end}" style="width:110px"></label></div>
    <textarea id="fixedNote" class="manual-note" placeholder="记录你听到的声音及判断依据">${esc(n.note||'')}</textarea><div class="controls"><button id="fixedSave">保存本段结论</button><button id="fixedExport">导出已保存的复核结论</button></div><p id="fixedSaved">仅保存人工记录，尚未自动修改生产数据。完整保留表示本段从头到尾；部分保留才使用上方起止时间。</p></div>
    <details><summary>转写、边界说明与下载</summary><p>${si}：${esc(r.text)}</p>${r.listener_events.map(e=>`<p>${li}：${fmt(e.start)}–${fmt(e.end)} ${esc(e.text)}</p>`).join('')}<p>边界提示：${esc(why(r.boundary_flags))||'无'}</p><p><a href="${esc(r.media)}" download>完整 pair 视频</a> · <a href="${esc(r.audio_urls.A)}" download>A 音轨</a> · <a href="${esc(r.audio_urls.B)}" download>B 音轨</a> · <a href="${esc(r.evidence_url)}" download>完整 QC 证据</a></p><p>沿用整轨响度归一化，无新增降噪或静音。视频帧／编码时长可能有不足一帧的量化差异，pair 元数据与 WAV 按源时间定义。</p></details>`;
  player=q('#fixedPlayer video');player.addEventListener('play',audio);player.addEventListener('timeupdate',draw);player.addEventListener('loadedmetadata',draw);player.addEventListener('seeked',draw);
  q('#fixedChannel').onchange=audio;q('#fixedReplay').onclick=()=>{player.currentTime=0;player.play().catch(showError)};
  q('#fixedHead').onclick=()=>{player.currentTime=0;player.play().catch(showError)};
  q('#fixedTail').onclick=()=>{player.currentTime=Math.max(0,r.duration-3);player.play().catch(showError)};
  q('#fixedSourceContext').onclick=()=>{player.pause();seek(Math.max(0,r.start-3),true);document.querySelector('.player').scrollIntoView({block:'center'});};
  q('#fixedSourceTail').onclick=()=>{player.pause();seek(Math.max(0,r.end-3),true);document.querySelector('.player').scrollIntoView({block:'center'});};
  q('#fixedStrip').onclick=e=>{const rect=e.target.getBoundingClientRect();player.currentTime=(e.clientX-rect.left)/rect.width*r.duration};
  q('#fixedPlayer').querySelectorAll('[data-evidence-time]').forEach(b=>b.onclick=()=>{player.currentTime=Math.max(0,Number(b.dataset.evidenceTime)-r.start-.2);player.play().catch(showError)});
  q('#fixedPlayer').querySelectorAll('[data-child]').forEach(b=>b.onclick=()=>select(b.dataset.child));
  q('#fixedPrev').onclick=()=>select(visible[Math.max(0,visible.findIndex(x=>x.id===r.id)-1)]?.id);
  q('#fixedNext').onclick=()=>select(visible[Math.min(visible.length-1,visible.findIndex(x=>x.id===r.id)+1)]?.id);
  q('#fixedSave').onclick=save;q('#fixedExport').onclick=exportFeedback;list();draw();
}
function overview(){
  const canvas=q('#fixedOverview'),g=canvas.getContext('2d'),w=canvas.width;
  g.fillStyle='#152239';g.fillRect(0,0,w,65);
  for(const r of set.rows.filter(x=>x.initial)){
    g.fillStyle=r.pool==='keep_candidate'?'#4b9b79':r.pool==='audio_reject'?'#d56e66':'#d79b49';
    g.fillRect(r.start/set.source_seconds*w,7,Math.max(1,r.duration/set.source_seconds*w),25);
  }
  g.fillStyle='#d9e6f5';g.font='15px sans-serif';g.fillText('00:00',5,58);g.fillText(fmt(set.source_seconds),w-115,58);
  canvas.onclick=e=>{const rect=e.target.getBoundingClientRect(),t=(e.clientX-rect.left)/rect.width*set.source_seconds;const r=set.rows.find(x=>x.initial&&x.start<=t&&t<x.end);if(r){q('#fixedFilter').value='all';filter();select(r.id)}};
}
function filter(){
  const mode=q('#fixedFilter').value;q('#fixedSemanticHelp').hidden=mode!=='semantic';
  visible=set.rows.filter(r=>mode==='terminal'?set.terminal_ids.includes(r.id):r.initial&&(mode==='all'||(mode==='semantic'?semanticReview(r):r.pool!=='keep_candidate')));
  if(visible.length)select(visible[0].id);else{stop();current=null;window.fixedPairCurrent=null;q('#fixedList').innerHTML='此类别没有片段';q('#fixedPlayer').innerHTML=''}
}
function intervalComparison(report){
  if(!report)return '';
  const cell=rows=>rows.length?rows.map(r=>`<p><strong>${esc(r.speaker)} · ${r.start.toFixed(6)}–${r.end.toFixed(6)} 秒</strong><br>${esc(r.text)}<br>${esc(poolNames[r.pool]||r.pool)}</p>`).join(''):'无最终通过片段';
  return `<details open id="${report.title?'productionIntervals':'adjacentIntervals'}"><summary>${esc(report.title||'统一 L→H：逐段起止时间与内容')}</summary><p>${esc(report.description||'L＝上一版局部反馈＋句末优先；H＝在 L 上增加短回应句首补接。时间均相对原视频，以秒计。初始候选与最终通过分别比较。')}</p>${report.stages.map(stage=>`<details ${stage.stage==='最终通过'?'open':''}><summary>${esc(stage.stage)}：${stage.before_count}→${stage.after_count} 段，${stage.unchanged} 段起止完全一致</summary><div style="overflow-x:auto"><table style="width:100%;table-layout:fixed"><thead><tr><th style="width:12%">起止变化</th><th>${esc(report.before_label||'L：起止／原文／状态')}</th><th>${esc(report.after_label||'H：起止／原文／状态')}</th></tr></thead><tbody>${stage.groups.map(g=>`<tr><td>${g.unchanged?'不变':'变化'}</td><td>${cell(g.before)}</td><td>${cell(g.after)}</td></tr>`).join('')}</tbody></table></div></details>`).join('')}</details>`;
}
function comparisonMarkup(set){
  if(!set.comparison_cases)return '';
  const row=(r,isNew)=>`<div class="comparison"><p>${r.id} · ${r.start.toFixed(6)}–${r.end.toFixed(6)} 秒 · ${r.duration.toFixed(3)} 秒 · ${poolNames[r.pool]||r.pool}</p><video controls playsinline preload="metadata" src="${esc(r.media)}" poster="${esc(r.poster)}"></video><p>${esc(r.text)}</p><button data-comparison-replay="${r.id}">播放 ${r.id}</button>${isNew?`<button data-new-candidate="${r.id}">查看本段证据／记录反馈</button>`:''}</div>`;
  const card=c=>`<article class="before-after" id="compare-${c.id}"><h3>${esc(c.title)}</h3><p class="compare-point">${esc(c.note)}</p><div class="before-after-grid ${c.production?'three-way':''}">${(c.production?['production','before','after']:['before','after']).map(side=>`<section><strong>${esc(side==='production'?(c.production_label||'旧全量：最终通过'):side==='before'?(c.before_label||set.comparison_before_label||'优化前（G 编号）'):(c.after_label||set.comparison_after_label||'优化后（L 编号）'))}</strong>${c[side].length?c[side].map(r=>row(r,side==='after')).join(''):'<p class="compare-point">此范围无最终通过片段</p>'}</section>`).join('')}</div>${c.excluded_proposals?.length?`<details class="excluded-proposals"><summary>单独查看未通过提案与原因（不计入右侧最终保留）</summary>${c.excluded_proposals.map(r=>`<p>${esc((r.reasons||[]).join('；'))}</p>${row(r,true)}`).join('')}</details>`:''}</article>`;
  const main=set.comparison_cases.filter(c=>!c.archived),archive=set.comparison_cases.filter(c=>c.archived);
  return `<details open id="localBcComparisons"><summary>${esc(set.comparison_title||'本次规则：处理前／处理后对照')}</summary><p>${esc(set.comparison_description||'左侧为上一版 5 秒切分，右侧为本次新候选。按各自实际边界播放，无额外上下文。')}</p>${main.map(card).join('')}${archive.length?`<details id="archivedComparisons"><summary>此前试验对照归档（不是本次旧全量对比）</summary>${archive.map(card).join('')}</details>`:''}</details>`;
}
window.renderFixedPairs=(revision)=>{
  document.querySelectorAll('#fixedPairReview video').forEach(v=>v.pause());stop();current=null;const versions=data.fixed_pair_versions||[data.fixed_pairs,data.fixed_pairs_previous].filter(Boolean);const urlParams=new URLSearchParams(location.search);const preferred=revision||urlParams.get('version')||versions.find(x=>x.rows.some(r=>r.id===urlParams.get('clip')))?.revision||versions.find(x=>x.comparison_cases?.some(c=>c.id===urlParams.get('compare')))?.revision;set=versions.find(x=>x.revision===preferred)||data.fixed_pairs||null;const section=document.querySelector('#fixedPairReview');section.hidden=!set;if(!set)return;section.open=new URLSearchParams(location.search).get('review')!=='1';
  const withheld=set.rows.filter(r=>r.initial&&r.pool!=='keep_candidate').length;
  section.querySelector('summary').textContent=` ${set.variant_title||'停顿阈值 '+(set.pause_threshold_seconds||3)+' 秒'} · ${set.initial_count} 段候选 · ${withheld} 段音质待复核`;
  const repaired=set.rows.filter(r=>r.initial&&(set.source_seconds<200?(r.boundary_repair&&r.start>=174&&r.end<=187):(r.start<=335.787&&r.end>=336.355)));
  const regrouped=set.rows.filter(r=>r.initial&&(set.pause_threshold_seconds||3)===5&&set.source_seconds<200&&!set.comparison_cases&&r.start>=118&&r.end<=146.1);
  q('#fixedBody').innerHTML=`<p><strong>先修复边界，再按原有音质规则筛选。</strong>${esc(set.scope_description||'这里只处理当前两条来源；每个播放器直接展示完整 pair，不加上下文。旧版未覆盖区间保留在下方历史区。')}</p>
    <div class="controls"><label>切分版本<select id="fixedVersion">${versions.map(x=>`<option value="${x.revision}" ${x.revision===set.revision?'selected':''}>${x.variant_title||(x.pause_threshold_seconds||3)+' 秒停顿阈值'}${x===data.fixed_pairs?'（当前）':'（历史）'}</option>`).join('')}</select></label><span>${esc(set.variant_description||'音质规则、2～11.95 秒时长约束不变。')}</span></div>
    ${set.seven_validation_note?`<p id="sevenValidationNote" class="compare-point"><strong>${esc(set.seven_validation_note)}</strong></p>`:''}
    ${intervalComparison(set.interval_comparison||set.adjacent_comparison)}
    ${comparisonMarkup(set)}
    ${set.holdout_sample?'':`<div class="note"><strong>前两处修复继续保留：</strong>${set.source_seconds<200?'短尾整句现已保留，前一段与尾段重新分配。':'句尾 “up.” 和倾听者 “Oh, yeah.” 现已包含在同一完整 pair 中。'}<div class="controls">${repaired.map(r=>`<button data-repaired="${r.id}">${r.id} · ${fmt(r.start)}–${fmt(r.end)} · ${poolNames[r.pool]}</button>`).join('')}</div></div>`}
    ${regrouped.length?`<div class="note"><strong>原 F1-13 附近已按 5 秒重切：</strong>“It was crazy” 现在位于同一候选中；其余句子边界仍按提示复核。<div class="controls">${regrouped.map(r=>`<button data-repaired="${r.id}">${r.id} · ${fmt(r.start)}–${fmt(r.end)} · ${poolNames[r.pool]}</button>`).join('')}</div></div>`:''}
    <p>初始候选 ${set.initial_count} 段：${Object.entries(set.initial_pools).map(([k,v])=>`${poolNames[k]||k} ${v} 段`).join('，')}。局部恢复后的规则通过覆盖 ${set.kept_seconds.toFixed(3)} 秒（旧生产结果 ${set.old_kept_seconds.toFixed(3)} 秒）。初始父 pair 与恢复子段不重复计入覆盖。另有 ${set.rows.filter(r=>r.initial&&semanticReview(r)).length} 段带语义边界提示，可单独筛选；与音质疑点可能重叠。</p>
    <canvas id="fixedOverview" width="1000" height="65" class="manual-strip"></canvas><p class="muted">修复后初始候选时间线：绿色＝规则通过，橙色＝待复核，红色＝规则排除；点击可选中完整 pair。空隙仍是未成段时间，不代表噪声。</p>
    <div class="controls"><select id="fixedFilter" aria-label="修复后 pair 筛选"><option value="withheld">音质待复核的候选（${withheld}）</option><option value="semantic">语义边界待复核（${set.rows.filter(r=>r.initial&&semanticReview(r)).length}）</option><option value="all">全部初始候选（${set.initial_count}）</option><option value="terminal">局部恢复后的终端 pair（${set.terminal_count}）</option></select>${(data.fixed_source_links||[{pair:"naturalistic_V02_S3579_I00000175",label:"190 秒来源"},{pair:"naturalistic_V03_S1909_I00000205",label:"354 秒来源"}]).map(x=>`<a href="?pair=${encodeURIComponent(x.pair)}&fixed=1#fixedPairReview">${esc(x.label)}</a>`).join(" ")}<span id="fixedProgress"></span></div>
    <p id="fixedSemanticHelp" class="note" hidden>此列表检查句首／句尾是否被拆开，不是音质排除清单。提示包括词边界片段、推测分句、长停顿和源转写首尾未确认；部分标签继承自所覆盖的原语句，不一定恰好对应当前切口，需结合首尾试听与原视频判断。黄色时间条仍表示音质疑点，与语义提示分开。</p>
    <div class="manual-layout"><aside id="fixedList" class="manual-list"></aside><section id="fixedPlayer" class="manual-player"></section></div>`;
  q('#fixedBody').querySelectorAll('[data-comparison-replay]').forEach(b=>b.onclick=()=>{const v=b.closest('.comparison').querySelector('video');v.currentTime=0;v.play().catch(showError)});
  q('#fixedBody').querySelectorAll('[data-new-candidate]').forEach(b=>b.onclick=()=>{q('#fixedFilter').value='all';filter();select(b.dataset.newCandidate);q('#fixedPlayer').scrollIntoView({block:'start'})});
  q('#fixedVersion').onchange=e=>window.renderFixedPairs(e.target.value);q('#fixedFilter').onchange=filter;q('#fixedFilter').value=set.default_filter||'withheld';filter();progress();overview();
  q('#fixedBody').querySelectorAll('[data-repaired]').forEach(b=>b.onclick=()=>{q('#fixedFilter').value='all';filter();select(b.dataset.repaired)});
  const params=new URLSearchParams(location.search);if(params.get('filter')==='semantic'){q('#fixedFilter').value='semantic';filter()}
  const requested=params.get('clip');if(requested&&set.rows.some(r=>r.id===requested)){if(!visible.some(r=>r.id===requested)){q('#fixedFilter').value='all';filter()}select(requested)}
  if(params.get('fixed')==='1'){section.open=true;requestAnimationFrame(()=>{const comparison=params.get('compare'),target=comparison?document.getElementById('compare-'+comparison):null;if(target){for(let p=target.parentElement;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true}}(target||section).scrollIntoView({block:'start'})})}
};
if(window.reviewData)window.renderFixedPairs();
})();
