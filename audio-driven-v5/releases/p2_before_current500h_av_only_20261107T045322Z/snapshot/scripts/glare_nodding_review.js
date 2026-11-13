/* Independent candidate channel. Never mutates VLM observations or training data. */
(() => {
  const $ = id => document.getElementById(id), esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const names={default:'默认',balanced:'平衡',recall:'高召回候选'}, key='glare-nodding50-review-v1';
  const reactionNames={nodding:'点头 · 已认可的高召回参数',head_shaking:'摇头',smiling:'微笑',laughing:'视觉大笑（不代表有笑声）',frowning:'皱眉',surprised:'惊讶样表情（不推断心理）'};
  const reactionKey='glare-reactions15-human-review-v1';let reactionSaved={};
  try {reactionSaved=JSON.parse(localStorage.getItem(reactionKey)||'{}');} catch (_) {}
  let data, current, visible=[], saved={};
  try {saved=JSON.parse(localStorage.getItem(key)||'{}');} catch (_) {}
  function store() {
    if(!current)return;
    saved[current.id]={sample_id:current.id,decision:$('decision').value,preferred:$('preferred').value,note:$('note').value,
      source_sha256:current.detector.video_sha256,protocol_fingerprint:current.detector.fingerprint,updated_at:new Date().toISOString()};
    try {localStorage.setItem(key,JSON.stringify(saved));$('reviewStatus').textContent='已保存在本浏览器；可导出 JSON。';} catch(e){$('reviewStatus').textContent='本地保存失败，请导出：'+e.message;}
  }
  function seek(a,b) {$('video').currentTime=Math.max(0,a-.35);$('video').play().catch(()=>{});}
  function renderVLM() {
    const idx=+$('window').value;
    $('vlm').innerHTML=['r4:video8','r4:video16'].map(k=>{
      const r=current.series[k][idx];if(!r)return '<p>该窗口缺失</p>';
      const events=Array.isArray(r.parsed)?r.parsed:(r.parsed?.events||[]);
      return `<section><h3>C · 原生视频 ${k.endsWith('16')?'16':'8'} fps</h3><span class="tag">${esc(r.status)} · ${r.start.toFixed(3)}–${r.end.toFixed(3)} 秒</span>`+
        events.map(e=>{const a=Number(e.start)+r.start,b=Number(e.end)+r.start;return `<div class="event"><button data-start="${a}" data-end="${b}">${a.toFixed(3)}–${b.toFixed(3)} 秒</button><div>${esc(e.behavior||e.description||JSON.stringify(e))}</div></div>`;}).join('')+
        `<details><summary>完整原始响应</summary><pre>${esc(r.raw)}</pre></details></section>`;
    }).join('');
  }
  function select(id) {
    current=data.samples.find(s=>s.id===id); if(!current)return;
    $('samples').value=id; $('video').pause();$('video').src=current.video;$('video').load();$('video').playbackRate=+$('speed').value;window.reviewAudioLoad(current);
    $('clipId').textContent=current.short+' · '+current.id;
    const d=current.detector;
    $('quality').textContent=d.status==='ok'?`人脸检出 ${d.detected_frames}/${d.frames} 帧；${(100*d.face_detection_fraction).toFixed(1)}%；视频 ${d.fps} fps，${current.duration.toFixed(3)} 秒。`:'检测失败：'+d.error;
    renderReactions();
    $('profiles').innerHTML=Object.entries(names).map(([k,name])=>{
      const events=d.profiles?.[k]?.events||[];
      return `<div class="card"><h3>${name}</h3><p>${d.status==='ok'?events.length+' 个候选区间':'检测失败，不能当作无点头'}</p>`+events.map((e,i)=>`<div class="event candidate" data-a="${e.start}" data-b="${e.end}"><button data-start="${e.start}" data-end="${e.end}">${e.start.toFixed(3)}–${e.end.toFixed(3)} 秒</button><div class="muted">候选 ${i+1} · 峰值 ${e.peak_score.toFixed(3)}${e.face_detection_fraction<1?' · 含缺失人脸帧':''}${e.touches_clip_boundary?' · 接触片段边界':''}</div></div>`).join('')+'</div>';
    }).join('');
    $('window').innerHTML=current.series['r4:video8'].map((r,i)=>`<option value="${i}">${esc(r.view)} · ${r.start.toFixed(2)}–${r.end.toFixed(2)} 秒</option>`).join('');renderVLM();
    const r=saved[id]||{};for(const f of ['decision','preferred','note'])$(f).value=r[f]||'';$('reviewStatus').textContent=r.updated_at?'已有本地复核记录':'';
    const u=new URL(location.href);u.searchParams.set('sample',id);history.replaceState(null,'',u);tick();
  }
  function renderReactions(){
    const r=current.reactions;$('reactions').hidden=!r;if(!r)return;
    const gate=r.headshake_direction_review;
    const intro='<div class="card"><h2>六类检测补充 · 允许同时发生</h2><p>点头沿用已认可的高召回参数；'+(gate?'摇头增加方向与反向运动检查，仍待复核；其余类别保持不变。':'其余使用参考代码默认参数，待本轮复核。')+'与下方 VLM 原文分别保留，不自动删除、改写或重复拼进生成 prompt。</p>'+(gate?'<p class="warning">摇头旧版存在混淆。新版只保留横向运动占主导、且有足够反向运动的候选；没有因“与点头重叠”直接删除。保留下来的仍可能是转头后回位，不等于已确认摇头。</p>':'')+'<p class="muted">视觉大笑可能与微笑重叠；皱眉／惊讶样表情是外观候选，不是情绪真值。六个分数没有共同概率尺度，不按最高分互相覆盖。</p>'+
      (r.status==='ok'?`<p class="muted">FER 检出 ${r.fer_detected.reduce((a,b)=>a+b,0)}/${r.frames} 帧；MediaPipe ${(100*r.face_detection_fraction).toFixed(1)}%。</p>${r.fer_detection_fraction<.9?'<p class="warning">FER 有较多帧未检出人脸，表情分数缺失。四类表情可能漏检或断裂，请结合画面和 VLM 核对；点头、摇头不依赖 FER。</p>':''}`:`<p>失败：${esc(r.error)}</p>`)+'</div>';
    $('reactions').innerHTML=intro+Object.entries(reactionNames).map(([k,name])=>{
      const events=r.channels?.[k]?.events||[],stored=reactionSaved[current.id+':'+k]||{},stale=k==='head_shaking'&&gate&&stored.source_fingerprint!==r.fingerprint,old=stale?{}:stored;
      const reasons={face_coverage:'人脸轨迹覆盖不足',yaw_amplitude:'横向幅度不足',yaw_range_dominant:'横向幅度未占主导',yaw_speed_dominant:'横向运动速度未占主导',opposite_yaw_excursions:'缺少足够幅度的反向运动'};
      const audit=k==='head_shaking'&&gate?'<details><summary>旧版摇头候选与本次筛除原因（'+gate.decisions.length+' 条）</summary>'+gate.decisions.map(d=>`<div class="event"><button data-start="${d.event.start}" data-end="${d.event.end}">${d.event.start.toFixed(3)}–${d.event.end.toFixed(3)} 秒</button><div>${d.retain?'保留为待复核候选':'暂不作为摇头补充：'+d.failed_checks.map(x=>reasons[x]).join('；')}</div><div class="muted">横向 ${d.metrics.yaw_range_deg.toFixed(2)}° / 纵向 ${d.metrics.pitch_range_deg.toFixed(2)}°；与点头重叠 ${d.nod_overlap_seconds.toFixed(2)} 秒（仅供核对）</div></div>`).join('')+'</details>':'';
      return `<section class="card reaction-card" data-reaction="${k}"><h3>${name} · ${r.status==='ok'?events.length+' 个候选':'失败'}</h3>`+
        (!events.length&&r.status==='ok'?'<p class="muted">本工具未检出，不等于确认该动作不存在。</p>':'')+
        events.map(e=>`<div class="event candidate" data-a="${e.start}" data-b="${e.end}"><button data-start="${e.start}" data-end="${e.end}">${e.start.toFixed(3)}–${e.end.toFixed(3)} 秒</button><span class="muted"> · 强度峰值 ${e.peak_score.toFixed(3)}${e.face_detection_fraction<1?' · 含 MediaPipe 缺失帧':''}${e.fer_detection_fraction<1?' · 含 FER 缺失帧':''}${e.touches_clip_boundary?' · 片段边界':''}</span></div>`).join('')+
        audit+`<label>此类复核 <select class="reaction-decision" data-type="${k}">${Object.entries({'':'未复核',correct:'基本正确',false_positive:'有误报',miss:'有漏检',both:'误报和漏检都有',absent:'未见该动作',uncertain:'不确定'}).map(([v,t])=>`<option value="${v}" ${old.decision===v?'selected':''}>${t}</option>`).join('')}</select></label><input class="reaction-note" data-type="${k}" placeholder="可选：时刻或问题" value="${esc(old.note||'')}" style="width:100%;margin-top:6px">${stale&&stored.updated_at?'<p class="muted">旧版摇头反馈仍保留；本版尚未复核。</p>':''}</section>`;
    }).join('');
  }
  function filter(preferred) {
    visible=data.samples.filter(s=>{const d=s.detector,p=d.profiles,any=p&&Object.values(p).some(x=>x.events.length),f=$('filter').value;if(f.startsWith('reaction:'))return !!s.reactions?.channels?.[f.split(':')[1]]?.events.length;return f==='all'||(f==='any'&&any)||(f==='added'&&any&&!p.default.events.length)||(f==='none'&&d.status==='ok'&&!any)||(f==='error'&&d.status!=='ok');});
    $('samples').innerHTML=visible.map(s=>`<option value="${s.id}">${s.short} · ${Object.keys(names).map(k=>s.detector.profiles?.[k]?.events.length??'失败').join(' / ')} · ${s.id}</option>`).join('');
    if(visible.length)select(visible.some(s=>s.id===preferred)?preferred:visible[0].id);
    else {$('video').pause();$('listenerAudio').pause();$('speakerAudio').pause();$('video').removeAttribute('src');$('video').load();$('profiles').textContent='此筛选无样本。';$('reactions').hidden=true;$('vlm').textContent='没有符合条件的片段。';$('clipId').textContent='此筛选无样本';$('quality').textContent='';$('pose').textContent='';$('clock').textContent='';$('window').innerHTML='';current=null;}
  }
  function tick(){if(!current)return;const t=$('video').currentTime,d=current.detector,i=Math.min(d.frames-1,Math.floor(t*(d.fps||25)));$('clock').textContent=`${t.toFixed(3)} / ${current.duration.toFixed(3)} 秒`; $('pose').textContent=d.status==='ok'?(d.pose.detected[i]?`当前帧 pitch ${d.pose.pitch[i].toFixed(2)}° · yaw ${d.pose.yaw[i].toFixed(2)}°`:'当前帧未检出人脸'):'';document.querySelectorAll('.candidate').forEach(e=>e.classList.toggle('on',t>=+e.dataset.a&&t<+e.dataset.b));}
  $('video').addEventListener('timeupdate',tick);$('speed').onchange=()=>{$('video').playbackRate=+$('speed').value;};
  document.addEventListener('click',e=>{const b=e.target.closest('[data-start]');if(b)seek(+b.dataset.start,+b.dataset.end);});
  document.addEventListener('input',e=>{if(!e.target.matches('.reaction-decision,.reaction-note'))return;const k=e.target.dataset.type,card=e.target.closest('.reaction-card');reactionSaved[current.id+':'+k]={sample_id:current.id,type:k,decision:card.querySelector('select').value,note:card.querySelector('input').value,source_fingerprint:current.reactions.fingerprint,updated_at:new Date().toISOString()};try{localStorage.setItem(reactionKey,JSON.stringify(reactionSaved));$('reviewStatus').textContent='分类复核已保存在本浏览器。';}catch(err){$('reviewStatus').textContent='保存失败，请导出：'+err.message;}});
  $('samples').onchange=()=>select($('samples').value);$('window').onchange=renderVLM;$('filter').onchange=()=>filter(current?.id);
  for(const [id,step] of [['prev',-1],['next',1]])$(id).onclick=()=>{if(visible.length)select(visible[(visible.findIndex(s=>s.id===current.id)+step+visible.length)%visible.length].id);};
  for(const f of ['decision','preferred','note'])$(f).addEventListener('input',store);
  $('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:data.reaction_summary?'glare-reactions-human-review-v1':'glare-nodding-human-review-v1',human_ground_truth:false,training_admitted:false,exported_at:new Date().toISOString(),records:Object.values(saved),reaction_records:Object.values(reactionSaved)},null,2)],{type:'application/json'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=data.reaction_summary?'glare_reactions_review.json':'glare_nodding_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
  fetch('data.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;window.glareReviewData=d;$('parameters').textContent=JSON.stringify({nod_profiles:d.profiles,other_detectors:d.reaction_protocol?.parameters,headshake_direction_gate:d.headshake_direction_protocol?.parameters},null,2);$('summary').textContent=`完成 ${d.summary.ok}/${d.summary.samples} 段；`+Object.entries(d.summary.profiles).map(([k,v])=>`${names[k]}：${v.clips_with_candidates} 段 / ${v.candidate_intervals} 个候选区间`).join('；');
    if(d.reaction_summary){document.title='GLARE 六类视觉补充 · 15 段';document.querySelector('h1').textContent='GLARE 六类视觉检测补充 · 15 段';$('summary').textContent=`六类检测 ${d.reaction_summary.ok}/${d.reaction_summary.samples} 段完成；`+Object.entries(d.reaction_summary.channels).map(([k,v])=>`${reactionNames[k]}：${v.clips} 段／${v.intervals} 区间`).join('；');$('nodComparison').open=false;document.querySelector('header .warning').textContent='点头高召回已获本轮人工认可；其他类别待复核。各类可同时发生，均单独标明来源；不是确定真值，不自动改写 VLM 原文。';
      for(const o of $('filter').options)if(['any','added','none'].includes(o.value))o.textContent='点头：'+o.textContent;
      for(const [k,n] of Object.entries(reactionNames)){const o=document.createElement('option');o.value='reaction:'+k;o.textContent='有候选：'+n;$('filter').append(o);}
      document.querySelector('.left .card:nth-child(2) h2').textContent='点头复核（历史反馈仍保留）';}
    if(d.selected_scheme){const p=document.createElement('p');p.className='warning';p.textContent='当前确认方案：原生视频 8 fps＋高召回点头补充。摇头已停用；其他四类专用检测仅作历史实验。下方保留历史结果供追溯，不进入当前默认输出。';document.querySelector('header h1').after(p);}
    filter(new URLSearchParams(location.search).get('sample'));}).catch(e=>{$('summary').textContent='加载失败：'+e.message;});
})();
