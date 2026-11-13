/* Shared numerical comparison component; uses frozen result-derived JSON. */
(function(global){
'use strict';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=v=>typeof v==='number'&&Number.isFinite(v);
function format(metric,value,delta=false){
 if(!finite(value))return delta?'—':'N/A';
 let n=value*(metric.percent?100:1);if(Math.abs(n)<1e-10)n=0;
 const sign=delta&&n>0?'+':'';
 return sign+n.toFixed(metric.digits)+(delta&&metric.percent?' pp':metric.percent?'%':delta&&metric.unit?' '+metric.unit:'');
}
function status(d){return d.delta===null?'na':Math.abs(d.delta)<1e-12||d.better===null?'flat':d.better?'good':'bad';}
function deltaCell(metric,d,extra=''){
 return '<td class="'+status(d)+'" data-metric="'+metric.key+'" data-raw-delta="'+(d.delta===null?'':d.delta)+'" '+extra+' title="'+esc('B−A；'+(metric.direction===0?'参考值，不单独判优':metric.direction>0?'越高越好':'越低越好')+(metric.percent?'；pp＝百分点':''))+'">'+format(metric,d.delta,true)+'</td>';
}
function render(target,doc,key){
 const ds=doc.datasets[key];if(!ds)throw Error('未知测试集 '+key);
 const byArm=Object.fromEntries(ds.rows.map(r=>[r.arm,r]));
 target.classList.add('p2-ablations');
 target.innerHTML=doc.groups.filter(group=>ds.comparisons.some(p=>p.group===group.id)).map(group=>{
  const pairs=ds.comparisons.filter(p=>p.group===group.id),metrics=doc.metrics.slice(0,8);
  const head='<thead><tr><th>对比：B−A</th>'+metrics.map(m=>'<th>'+esc(m.label)+' '+(m.direction>0?'↑':'↓')+' Δ'+(m.percent?' (pp)':'')+'</th>').join('')+'</tr></thead>';
  const rows=pairs.map(p=>'<tr data-comparison="'+p.id+'"><td><b>'+esc(p.label)+'</b><span class="p2-pair-label">A '+esc(byArm[p.baseline].label)+'<br>B '+esc(byArm[p.comparison].label)+'</span></td>'+metrics.map(m=>deltaCell(m,p.metrics.find(x=>x.key===m.key),'data-comparison="'+p.id+'"')).join('')+'</tr>').join('');
  const details=pairs.map(p=>'<details class="p2-original"><summary>'+esc(p.label)+'：展开原值、变化和音频指标</summary><p class="p2-muted">A：'+esc(byArm[p.baseline].label)+'<br>B：'+esc(byArm[p.comparison].label)+'</p><div class="p2-scroll"><table><thead><tr><th>指标</th><th>基线A</th><th>对照B</th><th>B−A</th><th>方向判断</th></tr></thead><tbody>'+doc.metrics.map(m=>{const d=p.metrics.find(x=>x.key===m.key);return '<tr><td>'+esc(m.label)+' '+(m.direction===0?'（参考）':m.direction>0?'↑':'↓')+'</td><td>'+format(m,d.baseline)+'</td><td>'+format(m,d.comparison)+'</td>'+deltaCell(m,d)+'<td class="'+status(d)+'">'+(d.delta===null?'无有效值':Math.abs(d.delta)<1e-12?'持平':d.better===null?'只作参考':d.better?'改善':'退化')+'</td></tr>';}).join('')+'</tbody></table></div></details>').join('');
  return '<section class="p2-ablation" id="ablation-'+group.id+'"><h3>'+esc(group.title)+'</h3><p class="p2-muted">'+esc(group.note)+'</p><div class="p2-scroll"><table class="p2-delta-table">'+head+'<tbody>'+rows+'</tbody></table></div>'+details+'</section>';
 }).join('');
 target.dataset.dataset=key;target.dataset.loaded='true';
}
async function mount(target,url,key){
 if(!target)return;
 try{const r=await fetch(url,{cache:'no-cache'});if(!r.ok)throw Error('数值数据HTTP '+r.status);const doc=await r.json();render(target,doc,key);}
 catch(e){target.textContent='消融数值加载失败：'+e.message;target.dataset.error='true';}
}
const style=document.createElement('style');style.textContent=`
.p2-ablations{min-width:0;font-size:13px;line-height:1.65}.p2-ablation{border:1px solid #dbe3ec;border-radius:10px;padding:16px;background:white;margin:14px 0;min-width:0;scroll-margin-top:18px}.p2-ablation h3{font-size:17px;margin:0 0 6px}.p2-muted{color:#607286;font-size:12px}.p2-scroll{max-width:100%;overflow:auto}.p2-ablations table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}.p2-ablations th,.p2-ablations td{padding:9px 10px;border-bottom:1px solid #e4eaf1;text-align:right}.p2-ablations th:first-child,.p2-ablations td:first-child{text-align:left}.p2-ablations th{background:#f1f5fa;color:#52677e}.p2-pair-label{display:block;font-size:11px;line-height:1.7;color:#62778c;white-space:normal;min-width:200px;max-width:260px}.p2-ablations td.good{color:#08735b;background:#eef8f3;font-weight:700}.p2-ablations td.bad{color:#a64b16;background:#fff3e9;font-weight:700}.p2-ablations td.flat,.p2-ablations td.na{color:#718096}.p2-original{border:1px solid #e3e9ef;border-radius:7px;padding:9px 12px;margin-top:10px}.p2-original summary{cursor:pointer;color:#315a7d;font-weight:600}.p2-original p{color:#607286}.p2-ablations .p2-delta-table td{font-variant-numeric:tabular-nums}@media(max-width:650px){.p2-ablation{padding:11px}.p2-ablation h3{font-size:16px}.p2-pair-label{min-width:170px;max-width:190px}}
`;document.head.appendChild(style);
global.P2Ablations={render,mount,format};
})(window);
