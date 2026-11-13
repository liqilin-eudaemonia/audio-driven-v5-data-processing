function renderMetrics(d){
 const mode=state.glareProtocol, selected=d.glare_protocols[mode],c=selected.coverage;
 $('#glare-protocol').disabled=false;$('#glare-protocol').value=mode;$('#metric-scope').disabled=true;$('#report').href='report.md';
 $('#metric-notes').innerHTML=`<p class="metric-coverage">六条件＋双seed共同支持：FVD ${c.fvd}/24 · UTMOS ${c.utmos}/24 · SyncNet ${c.sync}/24。每组48个输出全部评测完成。</p><p class="metric-warning">${mode==='unfiltered'?'GLARE去额外门槛：所有条件均使用全部24条。漏检沿用零信号、mask及插值规则，分数受检测误差影响。':'严格GLARE：表内另列各条件双seed共同成功数，条件间子集可能不同，不能直接排名。'} N/A不填0。全部输出都使用相同的全英文目标标注指导；旧4组仅更换Gemma输入，新2组指导同时进入Qwen与Gemma。训练数据也有变化，跨批结果不能单独归因于prompt。属于目标标注指导的诊断评测。</p>`;
 const keys=['FVD-32','UTMOSv2','LSE-D','LSE-C','R-F1','R-tIoU','R-ATD','R-FID'];
 $('#metric-rows').innerHTML=selected.metrics.map(r=>`<tr><td><strong>${esc(r.name)}</strong><span class="condition">${esc(r.variant)}</span><small>GLARE ${r.glare_samples}/24</small></td><td>${r.outputs}</td>${keys.map(k=>`<td title="${esc(r.scope)}">${fmt(r.values[k],k==='R-F1'?4:3)}</td>`).join('')}</tr>`).join('');
}
