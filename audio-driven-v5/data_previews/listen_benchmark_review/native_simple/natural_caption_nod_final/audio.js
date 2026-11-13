/* Independent participant audio, synchronized to the existing video master clock.
 * Frozen two-stage masks are applied to PCM in browser memory; no new WAV on disk.
 */
(() => {
  const byId = id => document.getElementById(id);
  const video = byId('video'), listener = byId('listenerAudio'), speaker = byId('speakerAudio');
  const mode = byId('audioMode'), note = byId('audioStatus');
  let current = null, epoch = 0, ready = false, objectURL = null;
  let controller = null;

  function inflate(spec, length) {
    const out = new Float32Array(length);
    for (const [a, b] of spec.ones) out.fill(1, a, b);
    for (const [start, encoded] of spec.fades) {
      const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
      const dv = new DataView(bytes.buffer);
      for (let i = 0; i < bytes.length / 4; i++) out[start + i] = dv.getFloat32(i * 4, true);
    }
    return out;
  }
  async function processedPCM(s, signal) {
    const response = await fetch(s.audio.original, {signal});
    if (!response.ok) throw Error('listener WAV HTTP ' + response.status);
    const raw = await response.arrayBuffer(), dv = new DataView(raw);
    const label = (a, n) => String.fromCharCode(...new Uint8Array(raw, a, n));
    if (label(0, 4) !== 'RIFF' || label(8, 4) !== 'WAVE') throw Error('不是 RIFF WAV');
    let format, bits, channels, rate, start, bytes;
    for (let p = 12; p + 8 <= raw.byteLength;) {
      const name = label(p, 4), n = dv.getUint32(p + 4, true), body = p + 8;
      if (body + n > raw.byteLength) throw Error('WAV 数据不完整');
      if (name === 'fmt ') { format = dv.getUint16(body, true); channels = dv.getUint16(body + 2, true); rate = dv.getUint32(body + 4, true); bits = dv.getUint16(body + 14, true); }
      if (name === 'data') { start = body; bytes = n; }
      p = body + n + (n % 2);
    }
    if (format !== 1 || ![16,24].includes(bits) || channels !== 1 || rate !== s.audio.sample_rate || bytes / (bits/8) !== s.audio.source_samples) throw Error('音频格式或采样长度与配方不符');
    // SHA256 source bytes were validated during build. Browser processing mirrors numpy's two float32 multiplies.
    const n = bytes / (bits/8), gate = inflate(s.audio.gate, n), extra = inflate(s.audio.extra, n), out = new Float32Array(n);
    const readSample = i => { if(bits===16) return dv.getInt16(start+2*i,true)/32768; const p=start+3*i; let v=dv.getUint8(p)|(dv.getUint8(p+1)<<8)|(dv.getUint8(p+2)<<16); if(v&0x800000)v-=0x1000000; return v/8388608; };
    for (let i = 0; i < n; i++) out[i] = Math.fround(Math.fround(readSample(i) * gate[i]) * extra[i]);
    return out;
  }
  function wavBlob(pcm, rate) {
    const bytes = new ArrayBuffer(44 + pcm.length * 4), dv = new DataView(bytes);
    const put = (a, s) => [...s].forEach((c, i) => dv.setUint8(a + i, c.charCodeAt(0)));
    put(0, 'RIFF'); dv.setUint32(4, 36 + pcm.length * 4, true); put(8, 'WAVE'); put(12, 'fmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 3, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, rate, true); dv.setUint32(28, rate * 4, true); dv.setUint16(32, 4, true); dv.setUint16(34, 32, true);
    put(36, 'data'); dv.setUint32(40, pcm.length * 4, true);
    for (let i = 0; i < pcm.length; i++) dv.setFloat32(44 + 4 * i, pcm[i], true);
    return new Blob([bytes], {type: 'audio/wav'});
  }
  const active = () => mode.value === 'dialogue' ? [listener, speaker] : mode.value === 'silent' ? [] : [listener];
  function volumes() {
    const value = +byId('audioVolume').value;
    listener.volume = value * (mode.value === 'dialogue' ? .5 : 1);
    speaker.volume = value * .5;
    listener.muted = false; speaker.muted = false;
  }
  function pauseAudio() { listener.pause(); speaker.pause(); }
  function sync(force = false) {
    for (const a of active()) {
      a.playbackRate = video.playbackRate;
      if (a.readyState && (force || Math.abs(a.currentTime - video.currentTime) > .08)) {
        a.currentTime = Math.min(video.currentTime, Number.isFinite(a.duration) ? a.duration : video.currentTime);
      }
    }
  }
  function startAudio() {
    if (!ready || video.paused) return;
    sync(true);
    for (const a of active()) a.play().catch(e => {
      if (e.name === 'AbortError' || video.paused) return;
      video.pause(); note.textContent = '浏览器未允许声音播放，请点击“有声播放／暂停”重试：' + e.message;
    });
  }
  function metadata(a, signal) {
    return new Promise((resolve, reject) => {
      const done = () => { cleanup(); resolve(); }, bad = () => { cleanup(); reject(Error('音轨加载失败')); }, aborted = () => { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); };
      const cleanup = () => { a.removeEventListener('loadedmetadata', done); a.removeEventListener('error', bad); signal.removeEventListener('abort', aborted); };
      a.addEventListener('loadedmetadata', done); a.addEventListener('error', bad); signal.addEventListener('abort', aborted, {once: true});
      if (signal.aborted) aborted(); else if (a.readyState >= 1) done();
    });
  }
  async function load(s) {
    const token = ++epoch;
    current = s; ready = false; pauseAudio(); controller?.abort(); controller = new AbortController();
    const signal = controller.signal;
    listener.removeAttribute('src'); speaker.removeAttribute('src'); listener.load(); speaker.load();
    if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
    if (mode.value === 'silent') { ready = true; note.textContent = '已选择静音；可切换上方音轨恢复声音。'; return; }
    note.textContent = '正在加载对应音轨…';
    try {
      if (mode.value === 'processed') {
        const pcm = await processedPCM(s, signal);
        if (token !== epoch) return;
        objectURL = URL.createObjectURL(wavBlob(pcm, s.audio.sample_rate));
        listener.src = objectURL;
      } else listener.src = s.audio.original;
      listener.load();
      const waits = [metadata(listener, signal)];
      if (mode.value === 'dialogue') { speaker.src = s.audio.speaker; speaker.load(); waits.push(metadata(speaker, signal)); }
      await Promise.all(waits);
      if (token !== epoch) return;
      ready = true; volumes(); sync(true);
      note.textContent = mode.value === 'processed'
        ? (s.audio.processed_full_silence ? '本段按现有后处理规则为全静音，不是播放故障。请切换原始 listener 或双人参考核对。' : '正在使用处理后 listener：官方词／句／VAD 并集＋多字中文静音，沿用训练配方。')
        : mode.value === 'dialogue' ? '双人参考：原始 listener 与 speaker 各 50% 音量；仅用于对话背景，不能把 speaker 声音当作 listener 回应。'
        : '原始 listener 独立音轨（可能有残留串音），未施加我们新增的静音规则。';
      if (!video.paused) startAudio();
    } catch (e) {
      if (token !== epoch || e.name === 'AbortError') return;
      pauseAudio(); video.pause(); note.textContent = '音频加载失败：' + e.message;
    }
  }
  mode.addEventListener('change', () => { if (current) load(current); });
  byId('audioVolume').addEventListener('input', volumes);
  byId('soundPlay').onclick = () => { if (video.paused) video.play().catch(e => note.textContent = e.message); else video.pause(); };
  video.addEventListener('play', startAudio);
  video.addEventListener('pause', pauseAudio);
  video.addEventListener('ended', pauseAudio);
  video.addEventListener('waiting', pauseAudio);
  video.addEventListener('playing', startAudio);
  video.addEventListener('seeking', () => { pauseAudio(); sync(true); });
  video.addEventListener('seeked', () => { sync(true); if (!video.paused) startAudio(); });
  video.addEventListener('ratechange', () => sync(true));
  video.addEventListener('timeupdate', () => sync());
  window.addEventListener('pagehide', pauseAudio);
  // Sample switches are explicit; listeners are paused before the next sample loads.
  window.reviewAudioLoad = load;
  window.reviewAudioState = () => ({ready, sample:current?.id, mode:mode.value, listenerTime:listener.currentTime, speakerTime:speaker.currentTime, listenerPaused:listener.paused, speakerPaused:speaker.paused});
  window.reviewProcessedPCM = processedPCM;
})();
