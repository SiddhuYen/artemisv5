'use strict';
/* ══════════════════════════════════════════════════════════════════════
   ARTEMIS console.

   Three views, one flow:

     CONSOLE  — set up the network, then run a connect
     NETWORK  — the imported roster
     RUN      — one job: live log, stats, and its routes with evidence

   Two rules shape everything below.

   1. A connect is always just POST /connect. There is no board to scope
      it to and no origin to tag; person_a and person_b are two names and
      the backend does the rest. The operator only ever *prefills* person_a.

   2. Nothing in this file infers anything. The backend returns spans,
      offsets, a resolution basis and an identity basis per hop, and the
      renderer's whole job is to show each of them as what it is — in
      particular to keep `resolved_statement` visibly separate from
      `span_text`, since one is derived and the other is the page's own
      words.
══════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let operator = null;      // {name, context} | null
let contacts = [];        // roster, newest fetch
let jobs = [];            // job summaries
let currentJobId = null;
let pollTimer = null;
let pickerTarget = 'a';

// ── toasts (V2 used alert(); a modal dialog per import is worse than this) ──
function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.textContent = message;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'err' ? 8000 : 4500);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 204) return null;
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (e) { body = text; }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : (typeof body === 'string' && body) || res.statusText;
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return body;
}

const json = (method, payload) => ({
  method,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(payload),
});

// ══════════════════════════════════════════════════════
// VIEWS
// ══════════════════════════════════════════════════════
function showView(id) {
  ['homeView', 'networkView', 'jobView'].forEach(v => $(v).classList.toggle('on', v === id));
  const scroll = $(id).querySelector('.hv-scroll');
  if (scroll) scroll.scrollTop = 0;
}

function showHome() {
  stopPolling();
  showView('homeView');
  loadJobs();
}

function showNetwork() {
  showView('networkView');
  renderNetwork();
}

// ══════════════════════════════════════════════════════
// OPERATOR
// ══════════════════════════════════════════════════════
function operatorInitials() {
  if (!operator) return '??';
  const parts = operator.name.trim().split(/\s+/);
  const initials = parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : parts[0].slice(0, 2);
  return initials.toUpperCase();
}

async function loadOperator() {
  try {
    operator = await api('/operator');
  } catch (e) {
    if (e.status !== 404) console.warn('operator load failed', e);
    operator = null;
  }
  renderOperator();
}

function renderOperator() {
  const name = operator ? operator.name.toUpperCase() : 'NO OPERATOR';
  ['hvUserName', 'nwUserName'].forEach(id => { $(id).textContent = name; });
  ['hvBadge', 'nwBadge'].forEach(id => { $(id).textContent = operatorInitials(); });
  $('hvUserRole').textContent = operator && operator.context ? '// ' + operator.context.toUpperCase() : '// SET';

  const v = $('stepOperatorV');
  if (operator) {
    v.classList.add('set');
    v.innerHTML = esc(operator.name) +
      (operator.context ? `<span class="sub">${esc(operator.context)}</span>` : '');
  } else {
    v.classList.remove('set');
    v.innerHTML = 'NOT SET<span class="sub">' +
      (contacts.length ? 'REQUIRED — A CSV IS LOADED' : 'OPTIONAL UNTIL A CSV IS LOADED') + '</span>';
  }
  // The operator is only load-bearing once there is a roster to own, so the
  // step is marked done when it is set OR when nothing needs it yet.
  $('stepOperator').classList.toggle('done', !!operator);
  $('stepOperator').classList.toggle('moot', !operator && contacts.length === 0);
  $('ccUseOperator').disabled = !operator;
  refreshFooter();
}

function editOperator() {
  $('opName').value = operator ? operator.name : '';
  $('opContext').value = operator ? operator.context || '' : '';
  $('opScrim').classList.add('open');
  setTimeout(() => $('opName').focus(), 30);
}
function closeOperator() { $('opScrim').classList.remove('open'); }
function opScrimClick(e) { if (e.target === $('opScrim')) closeOperator(); }

async function saveOperator() {
  const name = $('opName').value.trim();
  if (!name) { toast('An operator needs a name.', 'err'); return; }
  try {
    operator = await api('/operator', json('PUT', { name, context: $('opContext').value.trim() }));
    closeOperator();
    renderOperator();
    toast(`Operator set to ${operator.name}.`, 'ok');
  } catch (e) {
    toast('Could not save operator: ' + e.message, 'err');
  }
}

function useOperatorAsA() {
  if (!operator) return;
  $('ccPersonA').value = operator.name;
  if (operator.context && !$('ccContextA').value.trim()) $('ccContextA').value = operator.context;
}

// ══════════════════════════════════════════════════════
// NETWORK — LinkedIn CSV roster
// ══════════════════════════════════════════════════════
let pendingFile = null;
let pendingParsed = [];

async function loadContacts() {
  try {
    contacts = await api('/network/contacts');
  } catch (e) {
    console.warn('contacts load failed', e);
    contacts = [];
  }
  renderNetworkStats();
}

function renderNetworkStats() {
  const n = contacts.length;
  $('hvStatContacts').textContent = n.toLocaleString();
  $('nwCount').textContent = `[ ${n} ]`;
  $('nwFooterR').textContent = `${n} CONTACT${n === 1 ? '' : 'S'}`;
  const v = $('stepNetworkV');
  if (n) {
    v.classList.add('set');
    v.innerHTML = `${n.toLocaleString()} CONTACTS<span class="sub">LINKEDIN CONNECTIONS IMPORTED</span>`;
  } else {
    v.classList.remove('set');
    v.textContent = 'NO CSV LOADED';
  }
  $('stepNetwork').classList.toggle('done', n > 0);
  $('stepNetworkView').style.display = n ? '' : 'none';
  $('ccPickA').disabled = !n;
  $('ccPickB').disabled = !n;
  renderOperator();
}

function renderNetwork() {
  const q = ($('nwSearch').value || '').trim().toLowerCase();
  const rows = q
    ? contacts.filter(c => [c.name, c.role, c.company].join(' ').toLowerCase().includes(q))
    : contacts;
  const grid = $('nwGrid');
  if (!rows.length) {
    grid.innerHTML = `<div class="jv-empty" style="grid-column:1/-1">${
      contacts.length
        ? '// NO CONTACT MATCHES THAT FILTER'
        : '// NO CONTACTS YET — IMPORT YOUR LINKEDIN CONNECTIONS.CSV'
    }</div>`;
    return;
  }
  // Rendering 5,000 cards locks the tab for seconds and nobody scrolls that
  // far; the rest stay searchable, which is how they get found anyway.
  const shown = rows.slice(0, 300);
  grid.innerHTML = shown.map(c => {
    const sub = [c.role, c.company].filter(Boolean).join(' · ');
    return `<div class="nw-card">
      <div class="nw-name">${esc(c.name)}</div>
      <div class="nw-role">${esc(sub) || '&nbsp;'}</div>
      <div class="nw-actions">
        <button class="nw-btn" onclick="setEndpointFromNetwork('a', ${JSON.stringify(c.name).replace(/"/g, '&quot;')})">◧ AS A</button>
        <button class="nw-btn" onclick="setEndpointFromNetwork('b', ${JSON.stringify(c.name).replace(/"/g, '&quot;')})">◨ AS B</button>
      </div>
    </div>`;
  }).join('') + (rows.length > shown.length
    ? `<div class="jv-empty" style="grid-column:1/-1">// SHOWING ${shown.length} OF ${rows.length} — NARROW THE SEARCH TO SEE THE REST</div>`
    : '');
}

function setEndpointFromNetwork(side, name) {
  $(side === 'a' ? 'ccPersonA' : 'ccPersonB').value = name;
  const contact = contacts.find(c => c.name === name);
  const ctxEl = $(side === 'a' ? 'ccContextA' : 'ccContextB');
  // The role/company off the CSV is exactly the kind of short disambiguating
  // phrase context_a/context_b want. Never overwrite something already typed.
  if (contact && !ctxEl.value.trim()) {
    ctxEl.value = [contact.role, contact.company].filter(Boolean).join(' at ');
  }
  showHome();
  toast(`Person ${side.toUpperCase()} set to ${name}.`);
}

async function clearNetwork() {
  if (!confirm(`Delete all ${contacts.length} contacts? The operator stays set.`)) return;
  try {
    const out = await api('/network/contacts', { method: 'DELETE' });
    await loadContacts();
    renderNetwork();
    toast(`Deleted ${out.deleted} contacts.`, 'ok');
  } catch (e) {
    toast('Could not clear: ' + e.message, 'err');
  }
}

// ── import modal ──
function showImport() {
  pendingFile = null;
  pendingParsed = [];
  $('liPreview').style.display = 'none';
  $('liFileIn').value = '';
  $('liOperatorGate').style.display = operator ? 'none' : 'block';
  $('liOperatorName').value = '';
  $('liOperatorContext').value = '';
  refreshImportBtn();
  $('liScrim').classList.add('open');
}
function closeImport() { $('liScrim').classList.remove('open'); pendingFile = null; }
function liScrimClick(e) { if (e.target === $('liScrim')) closeImport(); }

function liDrop(e) {
  e.preventDefault();
  $('liDrop').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
}
function liFileChange(e) {
  const file = e.target.files[0];
  if (file) processFile(file);
}

function processFile(file) {
  pendingFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    pendingParsed = previewLinkedInCSV(String(ev.target.result));
    showPreview();
  };
  reader.readAsText(file);
}

/* A local parse purely so the drop zone can say what it is holding before
   anything is uploaded. The import itself POSTs the raw file — the server
   parse in artemis/network.py is the one that counts, and it handles the
   header-hunting and quoting cases this deliberately does not. */
function previewLinkedInCSV(text) {
  const rows = parseCSV(text);
  let headerIdx = -1;
  for (let i = 0; i < Math.min(20, rows.length); i++) {
    const folded = rows[i].map(c => c.trim().toLowerCase());
    if (folded.some(c => c.startsWith('first name')) && folded.some(c => c.startsWith('last name'))) {
      headerIdx = i; break;
    }
  }
  if (headerIdx === -1) return [];
  const headers = rows[headerIdx].map(h => h.trim().toLowerCase());
  const at = names => headers.findIndex(h => names.includes(h));
  const iFirst = at(['first name']), iLast = at(['last name']);
  const iRole = at(['position', 'title']), iCompany = at(['company', 'organization']);
  const cell = (row, i) => (i >= 0 && i < row.length ? row[i].trim() : '');
  const out = [];
  for (let i = headerIdx + 1; i < rows.length; i++) {
    const row = rows[i];
    if (!row.length || !row.some(c => c.trim())) continue;
    const name = `${cell(row, iFirst)} ${cell(row, iLast)}`.trim();
    if (name) out.push({ name, role: cell(row, iRole), company: cell(row, iCompany) });
  }
  return out;
}

/* Quote-aware, and aware that a quoted field can contain newlines — a
   LinkedIn headline routinely does. Splitting on /\r?\n/ first, the way V2
   did, tears those rows in half. */
function parseCSV(text) {
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else { inQuotes = false; }
      } else field += ch;
      continue;
    }
    if (ch === '"') inQuotes = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else if (ch !== '\r') field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function showPreview() {
  const known = new Set(contacts.map(c => c.name.toLowerCase()));
  const fresh = pendingParsed.filter(c => !known.has(c.name.toLowerCase()));
  const dupes = pendingParsed.length - fresh.length;
  $('liPreviewHdr').textContent = pendingParsed.length
    ? `${fresh.length} new · ${dupes} already imported`
    : 'no LinkedIn rows recognised — the server will try harder on upload';
  $('liPreviewRows').innerHTML = fresh.slice(0, 50).map(c => {
    const co = [c.role, c.company].filter(Boolean).join(' · ');
    return `<div class="li-preview-row">
      <div class="lpr-name">${esc(c.name)}</div>
      ${co ? `<div class="lpr-co">${esc(co)}</div>` : ''}
    </div>`;
  }).join('') + (fresh.length > 50
    ? `<div class="li-preview-row" style="color:var(--ink-faint)">…and ${fresh.length - 50} more</div>` : '');
  $('liPreview').style.display = 'block';
  refreshImportBtn();
}

function refreshImportBtn() {
  const needsOperator = !operator && !$('liOperatorName').value.trim();
  $('liImportBtn').disabled = !pendingFile || needsOperator;
}

async function confirmImport() {
  if (!pendingFile) return;
  const btn = $('liImportBtn');
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'IMPORTING…';
  try {
    // The upload is refused without an operator, so set one first when the
    // gate is showing — otherwise the user gets a 409 they already answered.
    if (!operator) {
      const name = $('liOperatorName').value.trim();
      if (!name) { toast('Set an operator to import.', 'err'); return; }
      operator = await api('/operator', json('PUT', { name, context: $('liOperatorContext').value.trim() }));
      renderOperator();
    }
    const form = new FormData();
    form.append('file', pendingFile);
    const out = await api('/network/upload', { method: 'POST', body: form });
    closeImport();
    await loadContacts();
    if ($('networkView').classList.contains('on')) renderNetwork();
    toast(`Imported ${out.created} new, ${out.updated} updated, ${out.skipped} unchanged — ${out.total} total.`, 'ok');
  } catch (e) {
    toast('Import failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = original;
    refreshImportBtn();
  }
}

// ── contact picker ──
function openPicker(side) {
  pickerTarget = side;
  $('cpTitle').textContent = `// PICK PERSON ${side.toUpperCase()}`;
  $('cpSearch').value = '';
  renderPicker();
  $('cpPanel').classList.add('open');
  setTimeout(() => $('cpSearch').focus(), 30);
}
function closePicker() { $('cpPanel').classList.remove('open'); }

function renderPicker() {
  const q = ($('cpSearch').value || '').trim().toLowerCase();
  const rows = (q ? contacts.filter(c => c.name.toLowerCase().includes(q)) : contacts).slice(0, 200);
  $('cpList').innerHTML = rows.length ? rows.map(c => {
    const sub = [c.role, c.company].filter(Boolean).join(' · ');
    const arg = JSON.stringify(c.name).replace(/"/g, '&quot;');
    return `<div class="cp-item dv-pick-item" onclick="pickContact(${arg})">
      <div class="cp-avatar">${esc(c.name.slice(0, 1).toUpperCase())}</div>
      <div class="cp-info">
        <div class="cp-name">${esc(c.name)}</div>
        <div class="cp-role">${esc(sub)}</div>
      </div>
    </div>`;
  }).join('') : `<div class="cp-empty">// NO MATCH</div>`;
}

function pickContact(name) {
  const side = pickerTarget;
  closePicker();
  setEndpointFromNetwork(side, name);
}

// ══════════════════════════════════════════════════════
// CONNECT — always just POST /connect
// ══════════════════════════════════════════════════════
async function runConnect() {
  const personA = $('ccPersonA').value.trim();
  const personB = $('ccPersonB').value.trim();
  if (!personA || !personB) { toast('Both people are required.', 'err'); return; }
  if (personA.toLowerCase() === personB.toLowerCase()) {
    toast('Person A and Person B are the same name.', 'err');
    return;
  }

  const depth = parseInt($('ccDepth').value, 10) || 2;
  const credits = parseInt($('ccCredits').value, 10);
  const wall = parseFloat($('ccWall').value);
  const budget = {};
  if (credits > 0) budget.max_serper_credits = credits;
  if (wall > 0) budget.wall_clock_s = wall;

  const payload = { person_a: personA, person_b: personB, max_depth: depth };
  const ctxA = $('ccContextA').value.trim();
  const ctxB = $('ccContextB').value.trim();
  if (ctxA) payload.context_a = ctxA;
  if (ctxB) payload.context_b = ctxB;
  if (Object.keys(budget).length) payload.budget = budget;

  const btn = $('ccRunBtn');
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = 'STARTING…';
  try {
    const accepted = await api('/connect', json('POST', payload));
    openJob(accepted.job_id, personA, personB);
  } catch (e) {
    toast('Connect failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

// ══════════════════════════════════════════════════════
// RUN VIEW
// ══════════════════════════════════════════════════════
function openJob(jobId, personA, personB) {
  stopPolling();
  currentJobId = jobId;
  shownLogLines = 0;
  $('jvA').textContent = personA || '—';
  $('jvB').textContent = personB || '—';
  $('jvJobId').textContent = jobId;
  $('jvLog').innerHTML = '';
  $('jvResult').innerHTML = '';
  $('jvWarnings').innerHTML = '';
  $('jvStats').innerHTML = '';
  $('jvFound').style.display = 'none';
  $('jvResultLbl').textContent = '// AWAITING RESULT';
  showView('jobView');
  startPolling();
}

let shownLogLines = 0;

function startPolling() {
  stopPolling();
  pollJob();
  pollTimer = setInterval(pollJob, 1200);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollJob() {
  if (!currentJobId) return;
  // A poll already in flight when the user opens a different run would
  // otherwise land afterwards and paint the old job's log and result into
  // the new job's view. Every path below re-checks the id it started with.
  const jobId = currentJobId;
  let job;
  try {
    job = await api(`/jobs/${jobId}`);
  } catch (e) {
    if (currentJobId !== jobId) return;
    stopPolling();
    $('jvResultLbl').textContent = '// LOST THE JOB';
    $('jvResult').innerHTML = `<div class="jv-empty">${esc(e.message)}</div>`;
    return;
  }
  if (currentJobId !== jobId) return;
  renderJobStatus(job);
  renderJobLog(job);
  renderJobStats(job);

  if (job.status === 'done' || job.status === 'failed') {
    stopPolling();
    $('jvCancelBtn').style.display = 'none';
    $('jvProgress').classList.remove('on');
    await renderResult(job);
    loadJobs();
  } else {
    $('jvCancelBtn').style.display = '';
    $('jvProgress').classList.add('on');
  }
}

function renderJobStatus(job) {
  const pill = $('jvStatus');
  pill.className = 'pill ' + job.status;
  pill.textContent = job.status.toUpperCase();
  const warnings = job.warnings || [];
  $('jvWarnings').innerHTML = warnings.length
    ? `<div class="rte-warn" style="border:1px solid oklch(0.62 0.15 75/0.4)">${
        warnings.map(w => `⚠ ${esc(w)}`).join('<br>')}</div>`
    : '';
}

function renderJobLog(job) {
  const log = job.log || [];
  if (log.length <= shownLogLines) return;
  const el = $('jvLog');
  const frag = document.createDocumentFragment();
  for (let i = shownLogLines; i < log.length; i++) {
    const entry = log[i];
    const line = document.createElement('div');
    line.className = 'jv-log-line l-' + (entry.level || 'info');
    const ev = document.createElement('span');
    ev.className = 'evt';
    ev.textContent = entry.event || '';
    const ms = document.createElement('span');
    ms.className = 'ms';
    ms.textContent = entry.message || '';
    line.append(ev, ms);
    frag.appendChild(line);
  }
  el.appendChild(frag);
  shownLogLines = log.length;
  el.scrollTop = el.scrollHeight;
}

const STAT_FIELDS = [
  ['serper_credits_used', 'SERPER CREDITS', true],
  ['serper_queries', 'QUERIES', false],
  ['pages_fetched', 'PAGES FETCHED', false],
  ['claude_calls', 'CLAUDE CALLS', false],
  ['nodes_expanded', 'NODES EXPANDED', false],
  ['merges', 'MERGES', false],
  ['merges_blocked', 'MERGES BLOCKED', false],
  ['elapsed_s', 'ELAPSED', true],
];

function renderJobStats(job) {
  const stats = job.stats || {};
  $('jvStats').innerHTML = STAT_FIELDS.map(([key, label, hot]) => {
    const raw = stats[key] ?? 0;
    const value = key === 'elapsed_s' ? `${Math.round(raw)}s` : Number(raw).toLocaleString();
    return `<div class="jv-stat${hot ? ' hot' : ''}"><span class="k">${label}</span><span class="v">${value}</span></div>`;
  }).join('');
}

async function cancelJob() {
  if (!currentJobId) return;
  try {
    await api(`/jobs/${currentJobId}`, { method: 'DELETE' });
    toast('Run cancelled.');
    pollJob();
  } catch (e) {
    toast('Could not cancel: ' + e.message, 'err');
  }
}

// ══════════════════════════════════════════════════════
// RESULTS — the evidence, shown as what it is
// ══════════════════════════════════════════════════════
async function renderResult(job) {
  if (job.status === 'failed') {
    $('jvResultLbl').textContent = '// FAILED';
    $('jvResult').innerHTML = `<div class="jv-empty">${esc(job.error || 'the job failed')}</div>`;
    return;
  }
  const result = job.result;
  if (!result) {
    $('jvResultLbl').textContent = '// NO RESULT';
    $('jvResult').innerHTML = `<div class="jv-empty">// THE JOB FINISHED WITHOUT A RESULT</div>`;
    return;
  }

  const found = $('jvFound');
  found.style.display = '';
  found.className = 'pill ' + (result.found ? 'done' : 'warn');
  found.textContent = result.found
    ? `${result.routes.length} ROUTE${result.routes.length === 1 ? '' : 'S'}`
    : 'NO ROUTE';

  if (result.disambiguation && result.disambiguation.length) {
    $('jvResultLbl').textContent = '// AMBIGUOUS ENDPOINT — PICK A READING AND RUN AGAIN';
    $('jvResult').innerHTML = renderDisambiguation(result.disambiguation);
    return;
  }

  if (!result.found || !result.routes.length) {
    $('jvResultLbl').textContent = '// NO ROUTE';
    $('jvResult').innerHTML = `<div class="jv-empty">
      // NO GROUNDED PATH WAS FOUND.<br>
      // EVERY REASON IS IN THE WARNINGS ABOVE AND THE LOG — A ROUTE IS NEVER OMITTED SILENTLY.
    </div>`;
    return;
  }

  // Name-match the people on the route against the roster. Purely a fact
  // about the CSV, reported beside the route; see NetworkStore.match.
  let inNetwork = {};
  if (contacts.length) {
    const names = new Set();
    result.routes.forEach(r => r.hops.forEach(h => { names.add(h.from.name); names.add(h.to.name); }));
    const openedFor = currentJobId;
    try {
      inNetwork = await api('/network/match', json('POST', { names: [...names].slice(0, 500) }));
    } catch (e) {
      console.warn('network match failed', e);
    }
    // The roster lookup is a second await; the user can have opened another
    // run across it, and these routes belong to the one they left.
    if (currentJobId !== openedFor) return;
  }

  $('jvResultLbl').textContent = `// ${result.routes.length} ROUTE${result.routes.length === 1 ? '' : 'S'} FOUND`;
  $('jvResult').innerHTML = result.routes.map((route, i) => renderRoute(route, i, inNetwork)).join('');
}

function renderDisambiguation(candidates) {
  return `<div class="rte"><div class="rte-hops">${candidates.map(c => {
    const attrs = Object.entries(c.attributes || {})
      .map(([k, vals]) => `${k}: ${(vals || []).join(', ')}`).filter(Boolean);
    return `<div class="hop first">
      <div class="hop-pip">○</div>
      <div class="hop-names"><span class="hop-name">${esc(c.display_name)}</span></div>
      ${attrs.length ? `<div class="ev"><div class="ev-k">ATTRIBUTES</div>
        <div class="ev-ctx">${esc(attrs.join(' · '))}</div></div>` : ''}
      ${c.snippet ? `<div class="ev"><div class="ev-k">SNIPPET</div>
        <div class="ev-ctx">${esc(c.snippet)}</div></div>` : ''}
      <div class="ev-meta">
        <a class="ev-src" href="${esc(c.source_url)}" target="_blank" rel="noopener noreferrer">${esc(c.source_title || c.source_url)} ↗</a>
      </div>
    </div>`;
  }).join('')}</div></div>`;
}

const BASIS_LABEL = {
  shared_page: 'SHARED PAGE',
  canonical_url: 'CANONICAL URL',
  attribute_match: 'ATTRIBUTE MATCH',
  name_only: 'NAME ONLY',
};

function identityChip(basis) {
  return `<span class="idb b-${esc(basis)}" title="How this observation was tied to the same human">
    ⛓ ${esc(BASIS_LABEL[basis] || basis.toUpperCase())}</span>`;
}

/* A fact about your CSV, not about this route's person. The label never
   claims the two are the same human — that is exactly the homonym error the
   identity layer exists to avoid, and a roster lookup is weaker evidence
   than anything on the merge ladder. */
function networkChip(name, inNetwork) {
  const hit = inNetwork[name];
  if (!hit) return '';
  const where = [hit.role, hit.company].filter(Boolean).join(' · ');
  let label, title;
  if (hit.basis === 'exact') {
    label = 'IN YOUR NETWORK';
    title = `"${hit.name}" is in your imported connections${where ? ` (${where})` : ''}. Name match only — not evidence that this is the same person.`;
  } else if (hit.basis === 'ambiguous') {
    label = 'NAME MATCHES ' + hit.candidates + ' CONTACTS';
    title = `${hit.candidates} people in your imported connections could go by this name. Which one, if either, this is cannot be told from the name.`;
  } else {
    label = 'MAYBE IN YOUR NETWORK';
    title = `Could be "${hit.name}" from your imported connections${where ? ` (${where})` : ''}. Name match only.`;
  }
  return `<span class="idb net" title="${esc(title)}">◈ ${esc(label)}</span>`;
}

function renderRoute(route, index, inNetwork) {
  const weakest = route.weakest_identity_basis;
  const warnings = route.identity_warnings || [];
  const hops = route.hops || [];

  const head = `<div class="rte-top">
    <span class="rte-n">ROUTE ${index + 1}</span>
    <span class="pill">${route.length} HOP${route.length === 1 ? '' : 'S'}</span>
    <span class="idb b-${esc(weakest)}" title="The weakest identity basis anywhere on this route">
      WEAKEST PIVOT · ${esc(BASIS_LABEL[weakest] || weakest.toUpperCase())}</span>
  </div>`;

  const warn = warnings.length
    ? `<div class="rte-warn">${warnings.map(w => `⚠ ${esc(w)}`).join('<br>')}</div>`
    : '';

  return `<div class="rte">${head}${warn}<div class="rte-hops">${
    hops.map((hop, i) => renderHop(hop, i, hops.length, inNetwork)).join('')
  }</div></div>`;
}

function renderHop(hop, i, total, inNetwork) {
  const coListing = hop.resolution_basis === 'co_listing';
  const classes = ['hop'];
  if (i === 0) classes.push('first');
  if (i === total - 1) classes.push('last');

  const retrieved = hop.retrieved_at ? String(hop.retrieved_at).replace('T', ' ').replace('Z', ' UTC') : '';

  return `<div class="${classes.join(' ')}">
    <div class="hop-pip">${i === 0 ? '●' : i === total - 1 ? '◆' : '○'}</div>

    <div class="hop-names">
      <span class="hop-name">${esc(hop.from.name)}</span>
      ${identityChip(hop.from_identity_basis)}
      ${networkChip(hop.from.name, inNetwork)}
      <span class="hop-to">──▶</span>
      <span class="hop-name">${esc(hop.to.name)}</span>
      ${identityChip(hop.to_identity_basis)}
      ${networkChip(hop.to.name, inNetwork)}
    </div>

    ${coListing ? `<div class="ev"><span class="idb co-listing" title="The page lists both people under one affiliation. Nobody asserted that they know each other.">
      ⓘ CO-LISTING — NOT A STATED RELATIONSHIP</span></div>` : ''}

    <div class="ev">
      <div class="ev-k">VERBATIM SPAN FROM THE PAGE</div>
      <div class="ev-span">“${esc(hop.span_text)}”</div>
    </div>

    ${hop.context_before ? `<div class="ev">
      <div class="ev-k">${coListing
        ? 'HEADING THAT ESTABLISHES THE SHARED AFFILIATION'
        : 'PRECEDING SENTENCES USED TO RESOLVE THE REFERENT'}</div>
      <div class="ev-ctx">${esc(hop.context_before)}</div>
    </div>` : ''}

    ${hop.resolved_statement && hop.resolved_statement !== hop.span_text ? `<div class="ev">
      <div class="ev-derived">
        <span class="tag">DERIVED ANNOTATION — NOT THE SOURCE'S WORDS</span>
        ${esc(hop.resolved_statement)}
      </div>
    </div>` : ''}

    <div class="ev-meta">
      <span class="idb" title="How the mention in the span was tied to a full name">
        ${esc(String(hop.resolution_basis).replace(/_/g, ' ').toUpperCase())}</span>
      <span class="ev-offsets">CHARS ${hop.span_start}–${hop.span_end}</span>
      ${retrieved ? `<span class="ev-offsets">RETRIEVED ${esc(retrieved)}</span>` : ''}
      <a class="ev-src" href="${esc(hop.source_url)}" target="_blank" rel="noopener noreferrer"
         title="${esc(hop.source_url)}">${esc(hop.source_title || hop.source_url)} ↗</a>
    </div>
  </div>`;
}

// ══════════════════════════════════════════════════════
// RUNS GRID
// ══════════════════════════════════════════════════════
async function loadJobs() {
  try {
    jobs = await api('/jobs?limit=60');
  } catch (e) {
    console.warn('jobs load failed', e);
    jobs = [];
  }
  renderJobs();
}

function renderJobs() {
  const q = ($('hvSearch').value || '').trim().toLowerCase();
  const rows = q
    ? jobs.filter(j => `${j.person_a} ${j.person_b}`.toLowerCase().includes(q))
    : jobs;
  $('hvStatJobs').textContent = jobs.length;
  $('hvJobCount').textContent = `[ ${rows.length} ]`;
  const grid = $('jobGrid');
  if (!rows.length) {
    grid.innerHTML = `<div class="jv-empty" style="grid-column:1/-1">${
      jobs.length ? '// NO RUN MATCHES THAT SEARCH' : '// NO RUNS YET — SET TWO PEOPLE ABOVE AND HIT RUN CONNECT'
    }</div>`;
    return;
  }
  grid.innerHTML = rows.map(j => {
    const outcome = j.status === 'done'
      ? (j.found ? `${j.routes} ROUTE${j.routes === 1 ? '' : 'S'}` : 'NO ROUTE')
      : j.status === 'failed' ? (j.error || 'FAILED') : '—';
    const stats = j.stats || {};
    return `<div class="hv-card fr" onclick="reopenJob(${JSON.stringify(j.id).replace(/"/g, '&quot;')})">
      <span class="br tl"></span><span class="br tr"></span>
      <span class="br bl"></span><span class="br br2"></span>
      <div class="hv-card-top">
        <span class="pill ${j.status}">${j.status.toUpperCase()}</span>
        <span class="hv-brd-id">${esc(String(j.created_at).slice(0, 16).replace('T', ' '))}</span>
      </div>
      <h3>${esc(j.person_a)} ◀▶ ${esc(j.person_b)}</h3>
      <div class="hv-card-stats">
        <div class="s"><span class="k">OUTCOME</span><span class="v">${esc(outcome)}</span></div>
        <div class="s hop"><span class="k">CREDITS</span><span class="v">${stats.serper_credits_used ?? 0}</span></div>
        <div class="s"><span class="k">PAGES</span><span class="v">${stats.pages_fetched ?? 0}</span></div>
        <span class="enter">OPEN ▸</span>
      </div>
    </div>`;
  }).join('');
}

function reopenJob(jobId) {
  const job = jobs.find(j => j.id === jobId);
  openJob(jobId, job ? job.person_a : '', job ? job.person_b : '');
}

// ══════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════
function refreshFooter() {
  $('hvFooterR').textContent =
    `${operator ? operator.name.toUpperCase() : 'NO OPERATOR'} · ${contacts.length} CONTACTS`;
}

async function checkHealth() {
  try {
    const health = await api('/health');
    $('hvHealth').textContent = health.serper_configured
      ? `LINK ESTABLISHED · ${health.extraction_model.toUpperCase()}`
      : 'NO SERPER KEY — CONNECT WILL BE REFUSED';
    $('hvDegraded').style.display = health.degraded ? '' : 'none';
  } catch (e) {
    $('hvHealth').textContent = 'BACKEND UNREACHABLE';
  }
}

document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  closeImport();
  closeOperator();
  closePicker();
});

(async function boot() {
  showView('homeView');
  await Promise.all([loadOperator(), loadContacts(), loadJobs(), checkHealth()]);
  renderNetworkStats();
})();
