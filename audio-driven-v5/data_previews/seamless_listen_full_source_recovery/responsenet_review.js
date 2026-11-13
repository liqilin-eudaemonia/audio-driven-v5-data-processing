/* 固定观察对象的审核页。ResponseNet 与 VidChatBench 共用，文案来自 page_copy。 */
function fixedSpeakerReview(data) {
  const kind = data && data.review_kind;
  return kind === 'responsenet_fixed_speaker' || kind === 'vidchatbench_fixed_speaker';
}

function responseNetRoleName(index) {
  if (fixedSpeakerReview(window.reviewData)) {
    return index ? 'Listener / Agent · 真实参考' : 'Speaker / Human · 观察对象';
  }
  return index ? 'B' : 'A';
}

function renderResponseNetReview() {
  const page = window.reviewPage;
  const data = window.reviewData;
  document.getElementById('cutMedia')?.remove();
  document.getElementById('officialAsr')?.remove();
  document.getElementById('originalMedia')?.remove();
  if (!page || !data || !fixedSpeakerReview(data)) return;
  const q = page.query;
  const esc = page.esc;
  const fmt = page.fmt;
  const summary = data.summary;
  const copy = data.page_copy || {};
  q('#method').textContent = copy.method || 'A 是 Speaker / Human · 观察对象，B 是 Listener / Agent。左声道 A、右声道 B，都是官方原音轨。橙色是这次切出的观察对象 pair，灰色是没有成 pair 的时间，包括 Agent 自己说话。音质模型没有跑。';
  document.querySelector('.legend').innerHTML = copy.legend || '<span><i class="dot" style="background:#ffb65d"></i>本次切出的观察对象 pair</span><span><i class="dot" style="background:#485975"></i>未成 pair（含 Agent 自己说话）</span><span><i class="dot" style="background:#de6e7a"></i>不足 2 秒等未成段</span><span>细线 = 每个 pair 的边界</span>';
  const cards = copy.cards || [
    ['原视频', `${summary.source_seconds.toFixed(1)} 秒`, '官方 speaker / listener 左右并排'],
    ['观察对象 pair', `${summary.final_kept_clips} 段 · ${summary.new_kept_seconds.toFixed(1)} 秒`, '未跑音质模型'],
    ['Agent 说话区间', `${summary.dropped_listener_regions} 个未反向切`, '不把听众当成说话人'],
    ['时间轴', '与原视频同一条', '点进度条直接跳到原视频'],
  ];
  q('#cards').innerHTML = cards.map(([title, value, note]) => `<div class="card">${title}<strong>${value}</strong><small>${note}</small></div>`).join('');
  q('#shotInfo').textContent = copy.shot || '没有做镜头检测。播放器里的是完整原时间线，切分不会把没选中的时间从原视频里删掉。';
  const player = document.querySelector('.player');
  if (data.original_media) {
    const box = document.createElement('section');
    box.id = 'originalMedia';
    box.className = 'panel';
    box.innerHTML = `<h2>${esc(copy.original_title || '24fps 本地片段')}</h2><p>${esc(copy.original_note || '')}</p><div class="comparisons">${data.original_media.map(item => `<div class="comparison"><strong>${esc(item.title)}</strong><video controls playsinline preload="metadata" src="${esc(item.file)}"></video><p>${item.seconds.toFixed(2)} 秒 · ${esc(item.note || '')}</p><p><a href="${esc(item.file)}" download>下载这条 24fps 片段</a></p></div>`).join('')}</div>`;
    player.insertAdjacentElement('beforebegin', box);
  }
  if (data.official_asr) {
    const asr = document.createElement('section');
    asr.id = 'officialAsr';
    asr.className = 'panel';
    asr.innerHTML = `<h2>${esc(copy.asr_title || 'ASR，speaker 和 Agent 分开')}</h2><p>${esc(copy.asr_note || '点一句会跳到原视频的对应时间。')}</p>` + data.official_asr.map(block => `<article><h3>${esc(block.title)}</h3><p>${esc(block.label || '')}</p><p>${esc(block.text || '这段没有词')}</p><div class="scroll"><table><tbody>${block.segments.map(row => `<tr data-time="${row.start}"><td>${fmt(row.start)}–${fmt(row.end)}${row.past_media ? ' · 超出本条音视频' : ''}</td><td>${esc(row.text)}</td></tr>`).join('')}</tbody></table></div></article>`).join('');
    player.insertAdjacentElement('beforebegin', asr);
    asr.querySelectorAll('tr[data-time]').forEach(row => row.onclick = () => page.seek(Number(row.dataset.time), false));
  }
  if (data.cut_media) {
    const box = document.createElement('section');
    box.id = 'cutMedia';
    box.className = 'panel';
    const roleName = {speaker: 'Speaker / Human · 观察对象', listener: 'Listener / Agent · 真实参考'};
    box.innerHTML = `<h2>${esc(copy.cut_title || '裁出来的 pair：画面和音轨分开')}</h2><p>${esc(copy.cut_note || '每条都按进度条上的起止时间，从官方原文件切开。视频里只有这一个人的画面，音轨是这个人自己的官方 wav。')}</p>` + data.cut_media.map(clip => `<article><h3>Pair ${clip.index} · ${fmt(clip.start)}–${fmt(clip.end)} · ${clip.duration_s.toFixed(2)} 秒</h3><p>${esc(clip.text)}</p><div class="comparisons">${['speaker', 'listener'].map(role => `<div class="comparison"><strong>${roleName[role]}</strong><video controls playsinline preload="metadata" src="${esc(clip.files[role].video)}"></video><audio controls preload="none" src="${esc(clip.files[role].audio)}"></audio>${clip.notes && clip.notes[role] ? `<p>${esc(clip.notes[role])}</p>` : ''}<p><a href="${esc(clip.files[role].video)}" download>下载画面</a> · <a href="${esc(clip.files[role].audio)}" download>下载音轨</a></p></div>`).join('')}</div></article>`).join('');
    document.getElementById('timeline').insertAdjacentElement('afterend', box);
  }
}

window.responseNetRoleName = responseNetRoleName;
window.renderResponseNetReview = renderResponseNetReview;
