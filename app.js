// Hidden Gems prototype. Everything runs in the browser: no server, no keys.
const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const COMMONS = 'https://commons.wikimedia.org/w/api.php';
const app = document.querySelector('#app');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { console.warn(e); return false; } },
  del(k) { try { localStorage.removeItem(k); } catch {} }
};

// ---------- state ----------
let cfg = { campus: { lat: 40.955, lon: -76.884, radius: 1200 }, model: 'Xenova/dinov2-small', search: [], landmarks: [], gems: [] };
let S = Object.assign({ radius: 60, thr: 0.55, useLoc: true, tester: true, matcher: 'dino' }, LS.get('hg_settings', {}));
let gems = [];
let P = LS.get('hg_progress', { idx: 0, fails: 0, log: [] });
const saveS = () => LS.set('hg_settings', S);
const saveP = () => LS.set('hg_progress', P);
const saveGems = () => { if (!LS.set('hg_gems', gems)) alert('Could not save gems on this device (storage full).'); };

// ---------- photo storage (IndexedDB) ----------
function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('hidden-gems', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('photos');
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function photoOp(mode, fn) {
  try {
    const db = await idb();
    return await new Promise((res, rej) => {
      const tx = db.transaction('photos', mode); const rq = fn(tx.objectStore('photos'));
      tx.oncomplete = () => res(rq && rq.result); tx.onerror = () => rej(tx.error);
    });
  } catch (e) { console.warn('photo storage', e); return null; }
}
const putPhoto = (id, blob) => photoOp('readwrite', s => s.put(blob, id));
const getPhoto = id => photoOp('readonly', s => s.get(id));
const clearPhotos = () => photoOp('readwrite', s => s.clear());

// ---------- images and embeddings ----------
async function toCanvas(blob, max) {
  let src;
  try { src = await createImageBitmap(blob); }
  catch {
    src = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = URL.createObjectURL(blob); });
  }
  const k = Math.min(1, max / Math.max(src.width, src.height));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(src.width * k)); c.height = Math.max(1, Math.round(src.height * k));
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  return c;
}
const canvasBlob = (c, q = 0.85) => new Promise(r => c.toBlob(r, 'image/jpeg', q));
async function shrink(blob, max) { return canvasBlob(await toCanvas(blob, max)); }
async function thumbData(blob) { return (await toCanvas(blob, 140)).toDataURL('image/jpeg', 0.6); }

let extractor = null, RawImage = null;
async function loadModel(onProgress) {
  if (S.matcher === 'basic' || extractor) return;
  let T;
  try { T = await import(TJS); } catch { throw new Error('Could not download the image model. Check your connection and try again.'); }
  T.env.allowLocalModels = false;
  RawImage = T.RawImage;
  extractor = await T.pipeline('image-feature-extraction', cfg.model, {
    dtype: 'q8',
    progress_callback: p => { if (onProgress && p.status === 'progress' && p.total) onProgress(Math.round(100 * p.loaded / p.total)); }
  });
}
function unit(v) { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return v.map(x => x / n); }
async function embed(blob) {
  let v;
  if (S.matcher === 'basic') {
    // Low-accuracy fallback: a 8x8 colour thumbnail. Only for checking app flow.
    const c = document.createElement('canvas'); c.width = c.height = 8;
    c.getContext('2d').drawImage(await toCanvas(blob, 64), 0, 0, 8, 8);
    const d = c.getContext('2d').getImageData(0, 0, 8, 8).data; v = [];
    for (let i = 0; i < d.length; i += 4) v.push(d[i], d[i + 1], d[i + 2]);
    const m = v.reduce((a, b) => a + b, 0) / v.length; v = v.map(x => x - m);
  } else {
    await loadModel();
    const img = await RawImage.fromBlob(await shrink(blob, 448));
    const out = await extractor(img);
    const dims = out.dims, data = out.data;
    if (dims.length === 3) {
      // [1, tokens, dim]: join the summary token with the average of the patch tokens.
      const n = dims[1], d = dims[2];
      const cls = Array.from(data.slice(0, d)); const mean = new Array(d).fill(0);
      for (let t = 1; t < n; t++) for (let j = 0; j < d; j++) mean[j] += data[t * d + j];
      v = unit(cls).concat(unit(mean));
    } else v = Array.from(data);
  }
  return unit(v).map(x => Math.round(x * 1e4) / 1e4);
}
const cosine = (a, b) => { if (!a || !b || a.length !== b.length) return -1; let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// ---------- location ----------
function metres(a, b) {
  const R = 6371000, r = x => x * Math.PI / 180;
  const h = Math.sin(r(b.lat - a.lat) / 2) ** 2 + Math.cos(r(a.lat)) * Math.cos(r(b.lat)) * Math.sin(r(b.lon - a.lon) / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
const locate = () => new Promise((res, rej) => {
  if (!navigator.geolocation) return rej(new Error('This browser has no location support.'));
  navigator.geolocation.getCurrentPosition(
    p => res({ lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy }),
    e => rej(new Error(e.code === 1 ? 'Location permission was denied.' : 'Could not get a location fix.')),
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 });
});

// ---------- build starter gems from Wikimedia Commons ----------
async function commons(params) {
  const pages = {}; let cont = {};
  for (let i = 0; i < 6; i++) {
    const q = new URLSearchParams({
      origin: '*', format: 'json', action: 'query', prop: 'imageinfo|coordinates', colimit: '50',
      iiprop: 'url|extmetadata|mime', iiurlwidth: '640', iiextmetadatafilter: 'LicenseShortName|Artist', ...params, ...cont
    });
    const j = await (await fetch(COMMONS + '?' + q)).json();
    for (const [id, p] of Object.entries(j.query?.pages || {})) pages[id] = Object.assign(pages[id] || {}, p);
    if (!j.continue) break; cont = j.continue;
  }
  return Object.values(pages);
}
const plain = html => new DOMParser().parseFromString(String(html || ''), 'text/html').body.textContent.trim().slice(0, 80);
async function buildFromWeb(status) {
  status('Looking for openly licensed campus photos…');
  const c = cfg.campus; let pages = [];
  const jobs = [commons({ generator: 'geosearch', ggscoord: `${c.lat}|${c.lon}`, ggsradius: String(c.radius), ggsnamespace: '6', ggslimit: '50' })];
  for (const s of cfg.search) jobs.push(commons({ generator: 'search', gsrsearch: s, gsrnamespace: '6', gsrlimit: '40' }));
  for (const r of await Promise.allSettled(jobs)) if (r.status === 'fulfilled') pages = pages.concat(r.value);
  if (!pages.length) throw new Error('Could not reach Wikimedia Commons.');
  const seen = new Set(), groups = [];
  for (const p of pages) {
    const ii = p.imageinfo?.[0]; if (!ii || seen.has(p.title)) continue; seen.add(p.title);
    const lic = ii.extmetadata?.LicenseShortName?.value || '';
    if (!/^(image\/jpeg|image\/png)$/.test(ii.mime || '')) continue;
    if (!/^(cc[ -]?by|cc0|public domain|pd)/i.test(lic) || /\b(nc|nd)\b/i.test(lic)) continue;
    if (/logo|seal|wordmark|\bmap\b|portrait|team|football|basketball game|player|aerial|18\d\d|19[0-6]\d/i.test(p.title)) continue;
    const title = p.title.replace(/^File:/, '').replace(/\.\w+$/, '').replace(/[_-]+/g, ' ');
    const lm = cfg.landmarks.find(l => l.match.some(m => title.toLowerCase().includes(m.toLowerCase())));
    const co = p.coordinates?.[0]; let pos = co ? { lat: co.lat, lon: co.lon } : null;
    if (pos && metres(pos, c) > c.radius * 1.5) pos = null;
    if (!pos && !lm) continue;
    const ref = { url: ii.thumburl || ii.url, page: ii.descriptionurl, license: lic, artist: plain(ii.extmetadata?.Artist?.value) || 'Unknown', title };
    let g = lm ? groups.find(x => x.lm === lm) : groups.find(x => !x.lm && metres(x, pos) < 35);
    if (!g) { g = lm ? { lm, lat: lm.lat, lon: lm.lon, name: lm.name, clue: lm.clue, refs: [] } : { ...pos, name: title, clue: 'Find this spot on campus: ' + title, refs: [] }; groups.push(g); }
    if (g.refs.length < 4) g.refs.push(ref);
  }
  groups.sort((a, b) => (b.lm ? 1 : 0) - (a.lm ? 1 : 0) || b.refs.length - a.refs.length);
  const picked = groups.slice(0, 12); const total = picked.reduce((n, g) => n + g.refs.length, 0); let done = 0;
  for (const g of picked) {
    const ok = [];
    for (const r of g.refs) {
      status(`Preparing reference photos… ${++done} of ${total}`);
      try { r.e = await embed(await (await fetch(r.url)).blob()); ok.push(r); } catch (e) { console.warn('ref failed', r.url, e); }
    }
    g.refs = ok;
  }
  // Walking order: start at the first gem, then always go to the nearest one left.
  const left = picked.filter(g => g.refs.length), out = [];
  while (left.length) {
    const from = out[out.length - 1]; let k = 0;
    if (from) left.forEach((g, i) => { if (metres(from, g) < metres(from, left[k])) k = i; });
    out.push(left.splice(k, 1)[0]);
  }
  if (!out.length) throw new Error('No usable openly licensed photos were found.');
  return out.map((g, i) => ({ id: 'web' + i, name: g.name, clue: g.clue, lat: g.lat, lon: g.lon, source: 'web', backupQ: '', backupA: '', refs: g.refs }));
}

// ---------- screens ----------
function view(html) { app.innerHTML = html; window.scrollTo(0, 0); }
const on = (sel, ev, fn) => { const el = app.querySelector(sel); if (el) el.addEventListener(ev, fn); };
const matcherNote = () => S.matcher === 'basic' ? '<p class="warn">Basic matcher is on. It is a low-accuracy stand-in for checking app flow only.</p>' : '';

async function home() {
  const n = gems.length, finished = n && P.idx >= n;
  view(`<h1>Hidden Gems</h1><p class="sub">The Best of Bucknell · prototype</p>
    <p>Follow each clue, take a photo at the spot, and the app checks you found the right place.</p>${matcherNote()}
    <div id="status" class="status"></div>
    ${n ? `<button class="big" id="play">${finished ? 'See your album' : P.idx ? `Continue (gem ${P.idx + 1} of ${n})` : `Start the hunt (${n} gems)`}</button>`
      : `<button class="big" id="build">Load starter gems from the web</button><p class="small">Pulls openly licensed campus photos and their coordinates from Wikimedia Commons. Needs a connection and takes a minute.</p>`}
    <div class="row"><button id="setup">Setup</button><button id="settings">Settings</button><button id="credits">Photo credits</button></div>`);
  on('#play', 'click', () => finished ? album() : play());
  on('#setup', 'click', setup); on('#settings', 'click', settings); on('#credits', 'click', credits);
  on('#build', 'click', async () => {
    const st = app.querySelector('#status'); app.querySelector('#build').disabled = true;
    try {
      if (S.matcher !== 'basic') { st.textContent = 'Downloading the image model…'; await loadModel(p => st.textContent = `Downloading the image model… ${p}%`); }
      gems = await buildFromWeb(t => st.textContent = t); saveGems(); P = { idx: 0, fails: 0, log: [] }; saveP(); home();
    } catch (e) {
      st.innerHTML = `<span class="bad">${esc(e.message || e)}</span><br>You can still add your own gems in Setup, or switch to the basic matcher in Settings if the model will not load.`;
      app.querySelector('#build').disabled = false;
    }
  });
}

function play() {
  const g = gems[P.idx]; if (!g) return album();
  const refs = S.tester ? `<details><summary>Tester: show reference photos</summary><div class="thumbs">${g.refs.map(r => `<img src="${esc(r.url || r.thumb || '')}" alt="">`).join('')}</div><p class="small">${g.lat.toFixed(5)}, ${g.lon.toFixed(5)}</p></details>` : '';
  view(`<p class="step">Gem ${P.idx + 1} of ${gems.length}</p><h2>${esc(S.tester ? g.name : 'Clue')}</h2>
    <p class="clue">${esc(g.clue)}</p>${refs}
    <label class="big filebtn">Take a photo<input id="cam" type="file" accept="image/*" capture="environment"></label>
    <label class="small filelink">or choose a saved photo<input id="lib" type="file" accept="image/*"></label>
    <div id="result" class="status"></div><div id="backup"></div>
    <div class="row"><button id="back">Home</button><button id="skipnow">Skip this gem</button></div>`);
  on('#back', 'click', home);
  on('#skipnow', 'click', () => { if (confirm('Skip this gem and go to the next clue?')) found(g, 'skipped', ''); });
  const handle = e => { const f = e.target.files[0]; e.target.value = ''; if (f) check(g, f); };
  on('#cam', 'change', handle); on('#lib', 'change', handle);
  if (P.fails >= 3) backup(g);
}

async function check(g, file) {
  const out = app.querySelector('#result'); out.textContent = 'Checking your photo…';
  try {
    const small = await shrink(file, 1024);
    const locP = S.useLoc ? locate().then(p => ({ p }), e => ({ err: e.message })) : Promise.resolve(null);
    if (S.matcher !== 'basic' && !extractor) { out.textContent = 'Downloading the image model…'; await loadModel(p => out.textContent = `Downloading the image model… ${p}%`); out.textContent = 'Checking your photo…'; }
    const [loc, e] = await Promise.all([locP, embed(small)]);
    const sim = Math.max(...g.refs.map(r => cosine(e, r.e)));
    const imgOK = sim >= S.thr;
    let locOK = true, locText = 'Location check is off.';
    if (loc) {
      if (loc.err) { locOK = false; locText = loc.err; }
      else { const d = metres(loc.p, g); locOK = d <= S.radius + Math.min(loc.p.acc || 0, 25); locText = `${Math.round(d)} m from the gem (limit ${S.radius} m, phone accuracy ±${Math.round(loc.p.acc || 0)} m).`; }
    }
    const detail = `<p class="small">Image similarity ${sim.toFixed(2)} (needs ${S.thr.toFixed(2)}) ${imgOK ? '✓' : '✗'}<br>${esc(locText)} ${loc ? (locOK ? '✓' : '✗') : ''}</p>`;
    P.log.push({ gem: g.id, sim: +sim.toFixed(3), locOK, imgOK, t: Date.now() });
    if (imgOK && locOK) { await putPhoto(g.id, small); return found(g, 'photo', detail); }
    P.fails++; saveP();
    const why = !imgOK && !locOK ? "That photo doesn't look like the gem, and you don't seem to be at the spot." : !imgOK ? "You're in the right area, but the photo doesn't look like the gem. Try another angle." : 'The photo looks right, but your phone says you are somewhere else.';
    out.innerHTML = `<p class="bad">${why}</p>${S.tester ? detail : ''}`;
    if (P.fails >= 3) backup(g, small);
  } catch (err) {
    console.error(err);
    out.innerHTML = `<p class="bad">Something went wrong checking that photo: ${esc(err.message || err)}</p>`;
  }
}

function backup(g, lastPhoto) {
  const el = app.querySelector('#backup'); if (!el) return;
  el.innerHTML = g.backupA
    ? `<div class="card"><p><b>Backup question</b><br>${esc(g.backupQ)}</p><input id="ans" placeholder="Your answer" autocomplete="off"><button id="go">Answer</button><p id="bmsg" class="small"></p></div>`
    : `<div class="card"><p>Three tries used. This gem has no backup question yet.</p><button id="skip">Move on to the next gem</button></div>`;
  const norm = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  on('#go', 'click', async () => {
    if (norm(app.querySelector('#ans').value) === norm(g.backupA)) { if (lastPhoto) await putPhoto(g.id, lastPhoto); found(g, 'backup', ''); }
    else app.querySelector('#bmsg').textContent = 'Not quite. Try again.';
  });
  on('#skip', 'click', async () => { if (lastPhoto) await putPhoto(g.id, lastPhoto); found(g, 'skipped', ''); });
}

function found(g, via, detail) {
  P.idx++; P.fails = 0; P.log.push({ gem: g.id, via, t: Date.now() }); saveP();
  const last = P.idx >= gems.length;
  view(`<p class="step">${via === 'skipped' ? 'Skipped' : 'Found it'}</p><h2>${esc(g.name)}</h2>
    <p class="good">${via === 'photo' ? 'Your photo matches.' : via === 'backup' ? 'Correct answer.' : 'Moving on.'}</p>${S.tester ? detail : ''}
    <button class="big" id="next">${last ? 'See your album' : 'Next clue'}</button>`);
  on('#next', 'click', () => last ? album() : play());
}

async function album() {
  view(`<h1>Your album</h1><p class="sub">The best of Bucknell, as you saw it.</p><div class="grid" id="grid"></div>
    <div class="row"><button id="home">Home</button><button id="again">Start over</button></div>`);
  on('#home', 'click', home);
  on('#again', 'click', async () => { if (!confirm('Erase your progress and photos?')) return; P = { idx: 0, fails: 0, log: [] }; saveP(); await clearPhotos(); home(); });
  const grid = app.querySelector('#grid');
  for (const g of gems) {
    const b = await getPhoto(g.id); const f = document.createElement('figure');
    f.innerHTML = (b ? `<img src="${URL.createObjectURL(b)}" alt="">` : '<div class="nophoto">No photo</div>') + `<figcaption>${esc(g.name)}</figcaption>`;
    grid.appendChild(f);
  }
}

function credits() {
  const rows = gems.flatMap(g => g.refs.filter(r => r.page).map(r => `<li><a href="${esc(r.page)}" target="_blank" rel="noopener">${esc(r.title)}</a> by ${esc(r.artist)}, ${esc(r.license)}, via Wikimedia Commons</li>`));
  view(`<h1>Photo credits</h1>${rows.length ? `<ul class="small">${rows.join('')}</ul>` : '<p>No web photos are in use.</p>'}
    <p class="small">Image model: DINOv2 (Meta), run with Transformers.js.</p><div class="row"><button id="home">Home</button></div>`);
  on('#home', 'click', home);
}

function settings() {
  view(`<h1>Settings</h1>
    <label>Distance limit: <b id="rv">${S.radius}</b> m<input id="radius" type="range" min="20" max="200" step="5" value="${S.radius}"></label>
    <label>Image similarity needed: <b id="tv">${S.thr.toFixed(2)}</b><input id="thr" type="range" min="0.2" max="0.9" step="0.01" value="${S.thr}"></label>
    <label class="check"><input id="useLoc" type="checkbox" ${S.useLoc ? 'checked' : ''}> Check location</label>
    <label class="check"><input id="tester" type="checkbox" ${S.tester ? 'checked' : ''}> Tester mode (show scores, names and reference photos)</label>
    <label class="check"><input id="basic" type="checkbox" ${S.matcher === 'basic' ? 'checked' : ''}> Use the basic matcher (low accuracy, no model download)</label>
    <p class="small">Changing the matcher makes saved reference photos unusable. Rebuild or re-record gems afterwards.</p>
    <div class="row"><button id="log">Download test log</button><button id="rebuild">Erase gems and progress</button></div>
    <div class="row"><button id="home">Done</button></div>`);
  on('#radius', 'input', e => { S.radius = +e.target.value; app.querySelector('#rv').textContent = S.radius; saveS(); });
  on('#thr', 'input', e => { S.thr = +e.target.value; app.querySelector('#tv').textContent = S.thr.toFixed(2); saveS(); });
  on('#useLoc', 'change', e => { S.useLoc = e.target.checked; saveS(); });
  on('#tester', 'change', e => { S.tester = e.target.checked; saveS(); });
  on('#basic', 'change', e => { S.matcher = e.target.checked ? 'basic' : 'dino'; saveS(); });
  on('#log', 'click', () => download('hunt-log.json', JSON.stringify({ settings: S, log: P.log }, null, 1)));
  on('#rebuild', 'click', async () => { if (!confirm('Erase all gems, progress and photos on this device?')) return; LS.del('hg_gems'); gems = (cfg.gems || []).slice(); P = { idx: 0, fails: 0, log: [] }; saveP(); await clearPhotos(); home(); });
  on('#home', 'click', home);
}

function download(name, text) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' })); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

let draft = null;
function setup() {
  draft = draft || { name: '', clue: '', backupQ: '', backupA: '', lat: null, lon: null, refs: [] };
  view(`<h1>Setup</h1><p class="small">Stand at a gem, record its location, and take 3 to 5 reference photos from different angles.</p>${matcherNote()}
    <ol class="gemlist">${gems.map((g, i) => `<li><b>${esc(g.name)}</b> <span class="small">${g.refs.length} photos · ${g.lat.toFixed(5)}, ${g.lon.toFixed(5)}</span>
      <span class="mini"><button data-up="${i}" ${i ? '' : 'disabled'}>↑</button><button data-del="${i}">Delete</button></span></li>`).join('') || '<li class="small">No gems yet.</li>'}</ol>
    <div class="card"><h2>New gem</h2>
      <input id="name" placeholder="Name (players see it after finding the gem)" value="${esc(draft.name)}">
      <textarea id="clue" rows="3" placeholder="Clue">${esc(draft.clue)}</textarea>
      <input id="bq" placeholder="Backup question" value="${esc(draft.backupQ)}">
      <input id="ba" placeholder="Backup answer" value="${esc(draft.backupA)}">
      <button id="loc">Record my location</button> <span id="locv" class="small">${draft.lat ? `${draft.lat.toFixed(5)}, ${draft.lon.toFixed(5)}` : 'Not recorded'}</span>
      <label class="filebtn">Add reference photos<input id="refs" type="file" accept="image/*" multiple></label>
      <div class="thumbs">${draft.refs.map(r => `<img src="${r.thumb}" alt="">`).join('')}</div>
      <p id="smsg" class="status"></p><button class="big" id="save">Save gem</button></div>
    <div class="row"><button id="export">Export gems.json</button><button id="home">Home</button></div>
    <p class="small">Export, then replace gems.json in your repository so every player gets your gems.</p>`);
  const keep = () => { draft.name = app.querySelector('#name').value; draft.clue = app.querySelector('#clue').value; draft.backupQ = app.querySelector('#bq').value; draft.backupA = app.querySelector('#ba').value; };
  const msg = t => { const m = app.querySelector('#smsg'); if (m) m.textContent = t; };
  app.querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', () => { if (confirm('Delete this gem?')) { gems.splice(+b.dataset.del, 1); saveGems(); keep(); setup(); } }));
  app.querySelectorAll('[data-up]').forEach(b => b.addEventListener('click', () => { const i = +b.dataset.up; [gems[i - 1], gems[i]] = [gems[i], gems[i - 1]]; saveGems(); keep(); setup(); }));
  on('#loc', 'click', async () => { keep(); msg('Getting your location…'); try { const p = await locate(); draft.lat = p.lat; draft.lon = p.lon; setup(); app.querySelector('#smsg').textContent = `Recorded (phone accuracy ±${Math.round(p.acc)} m).`; } catch (e) { msg(e.message); } });
  on('#refs', 'change', async e => {
    keep(); const files = [...e.target.files];
    try {
      if (S.matcher !== 'basic' && !extractor) { msg('Downloading the image model…'); await loadModel(p => msg(`Downloading the image model… ${p}%`)); }
      for (let i = 0; i < files.length; i++) { msg(`Processing photo ${i + 1} of ${files.length}…`); draft.refs.push({ e: await embed(files[i]), thumb: await thumbData(files[i]) }); }
      setup();
    } catch (err) { console.error(err); msg('Could not process photos: ' + (err.message || err)); }
  });
  on('#save', 'click', () => {
    keep();
    if (!draft.name || !draft.clue) return msg('Add a name and a clue.');
    if (draft.lat == null) return msg('Record the location first.');
    if (!draft.refs.length) return msg('Add at least one reference photo.');
    gems.push({ id: 'g' + Date.now(), source: 'setup', ...draft }); saveGems(); draft = null; setup();
  });
  on('#export', 'click', () => download('gems.json', JSON.stringify({ ...cfg, gems }, null, 0)));
  on('#home', 'click', () => { keep(); home(); });
}

// ---------- start ----------
(async () => {
  try { cfg = Object.assign(cfg, await (await fetch('gems.json', { cache: 'no-cache' })).json()); } catch (e) { console.warn('gems.json not loaded', e); }
  gems = LS.get('hg_gems', null) || (cfg.gems || []).slice();
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
  home();
})();
