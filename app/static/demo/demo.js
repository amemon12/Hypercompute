'use strict';

// This page rehearses the workflow. Only the web join is shared across browsers.
const DEVICES = [
  {id:'gx10', name:'GX10', chip:'NVIDIA GB10', memory:121.6, detail:'128 GB unified memory', role:'Host', color:'#9bb7c6', layers:56},
  {id:'rtx', name:'GeForce RTX', chip:'5070 Ti Laptop GPU', memory:11.9, detail:'12 GB GPU memory', role:'This computer', color:'#b38d6d', layers:3},
  {id:'mac', name:'MacBook Air', chip:'Apple M4', memory:16, detail:'16 GB unified memory', role:'Friend', color:'#718d68', layers:2},
];
const TOTAL_LAYERS = 61;
const $ = id => document.getElementById(id);
const setText = (id, text) => {if ($(id).textContent !== text) $(id).textContent = text;};
const guest = location.pathname.replace(/\/$/, '') === '/join';
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let hostToken = new URLSearchParams(location.hash.slice(1)).get('t') || '';
try {
  if (hostToken) sessionStorage.setItem('hc-demo-host', hostToken);
  else hostToken = sessionStorage.getItem('hc-demo-host') || '';
} catch (_) { /* The launch token still works for this tab if storage is unavailable. */ }
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
let state = {hosting:false, joined:false, version:-1, session:null, code:null};
let phase = 'off', epoch = 0, lastVersion = -1, instance = null;
let connected = false, pollBusy = false, busyToggle = false, chatBusy = false, joinBusy = false;
let copyNotice = '', copyError = '', joinError = '';
const retired = new Set(), timers = new Map();
const layerNodes = Array.from({length:TOTAL_LAYERS}, () => {
  const node = document.createElement('i'); node.setAttribute('aria-hidden','true'); $('layers').append(node); return node;
});
const laptop = '<svg viewBox="0 0 120 94" aria-hidden="true"><rect class="case" x="15" y="21" width="90" height="62" rx="3"/><rect class="screen" x="20" y="26" width="80" height="52" rx="1"/><path class="case" d="M5 84h110v3c0 4-6 6-11 6H16C11 93 5 91 5 87z"/><path d="M47 84h26l-3 3H50z"/></svg>';
const desktop = '<svg viewBox="0 0 120 94" aria-hidden="true"><rect class="case" x="30" y="10" width="60" height="76" rx="3"/><path d="M40 19v58M48 23h31m-31 7h31m-31 7h31"/><rect class="screen" x="48" y="45" width="31" height="22" rx="1"/><circle cx="76" cy="76" r="2"/></svg>';
DEVICES.forEach(device => {
  const node = document.createElement('div'); node.className = 'computer'; node.id = `computer-${device.id}`;
  node.style.setProperty('--device-color', device.color);
  node.innerHTML = `<div class="device-art">${device.id === 'gx10' ? desktop : laptop}</div><h3></h3><p class="chip"></p><p class="spec"></p><span class="device-role"></span>`;
  node.querySelector('h3').textContent = device.name;
  $('computers').append(node);
});
function wait(ms, run = epoch) {
  return new Promise(resolve => {
    const timer = setTimeout(() => {timers.delete(timer); resolve(run === epoch);}, ms);
    timers.set(timer, resolve);
  });
}
function cancel() {
  epoch++;
  timers.forEach((resolve, timer) => {clearTimeout(timer); resolve(false);}); timers.clear();
  chatBusy = false; copyNotice = ''; copyError = ''; joinError = '';
  $('response').hidden = true; $('response').removeAttribute('aria-busy'); $('answer').textContent = '';
  layerNodes.forEach(node => node.style.background = '');
  $('layers').setAttribute('aria-label', '61 layers, not allocated');
  $('host-layers').textContent = 'Layer allocation appears here'; $('friend-layers').textContent = '';
  $('layer-legend').classList.remove('allocated');
}
async function api(path, body) {
  const headers = {'Content-Type':'application/json'};
  if (!guest && hostToken) headers['X-Demo-Host'] = hostToken;
  const response = await fetch(path, {method:body === undefined ? 'GET' : 'POST', headers, cache:'no-store',
    body:body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The session is unavailable. Try again.');
  return data;
}
function adopt(next) {
  if (next.instance && next.instance !== instance) {
    if (retired.has(next.instance)) return;
    if (instance) retired.add(instance);
    instance = next.instance; lastVersion = -1; cancel(); phase = 'off';
    state = {hosting:false, joined:false, session:null};
  }
  if (next.version < lastVersion) return;
  lastVersion = next.version; connected = true; $('connection-error').hidden = true;
  const fresh = next.hosting !== state.hosting || (!guest && next.session !== state.session);
  const arrival = next.joined && (!state.joined || fresh);
  state = next;
  if (fresh) {cancel(); phase = state.hosting ? 'waiting' : 'off';}
  render();
  if (arrival && !guest) allocate();
}
function render() {
  if (guest) {
    $('join-button').disabled = !connected || !state.hosting || state.joined || joinBusy || $('code-input').value.trim().length !== 6;
    $('code-input').disabled = state.joined || joinBusy;
    $('join-button').innerHTML = state.joined ? 'Joined ✓' : 'Join session <span aria-hidden="true">↗</span>';
    setText('join-state', joinError || (joinBusy ? 'Joining the session…' : !connected ? 'Connecting to the session…' : !state.hosting ? 'The host has not started a session yet.' : state.joined ? 'Your MacBook is connected. The host is assigning the layers.' : 'Enter the six-character code your friend shared.'));
    return;
  }
  const on = state.hosting, joined = on && state.joined;
  const ready = on && phase === 'ready';
  const calculated = ['splitting','loading','ready'].includes(phase);
  $('hosting').setAttribute('aria-checked', String(on)); $('hosting').disabled = !connected || busyToggle;
  setText('hosting-label', busyToggle ? 'Wait' : on ? 'On' : 'Off');
  setText('hosting-help', on ? 'Session open. Invite your friend to join.' : 'Turn on hosting to get started.');
  $('copy-link').disabled = !on || !connected;
  $('copy-link').querySelector('span').textContent = copyNotice ? 'Copied' : 'Copy invite';
  $('join-code').hidden = !on; setText('join-code', state.code || '');
  setText('invite-help', !on ? 'Available when hosting is on.' : copyError || copyNotice || 'Share this code and the join link.');
  setText('pool-state', !on ? 'Not connected' : joined ? '3 computers connected' : '2 computers / waiting for a friend');
  DEVICES.forEach(device => {
    const node = $(`computer-${device.id}`), active = on && (device.id !== 'mac' || joined);
    node.classList.toggle('connected', active);
    node.querySelector('.chip').textContent = active ? device.chip : device.id === 'mac' ? 'Waiting for a friend' : device.chip;
    node.querySelector('.spec').textContent = !active ? device.id === 'mac' ? 'Join with a code' : device.detail : device.id === 'mac' && phase === 'reading' ? 'Reading demo specifications…' : device.detail;
    node.querySelector('.device-role').textContent = device.role;
  });
  const members = DEVICES.filter(device => on && (device.id !== 'mac' || calculated));
  const total = members.reduce((sum, device) => sum + device.memory, 0);
  setText('memory-total', total ? total.toFixed(1) : '0');
  $('memory-track').replaceChildren(); $('memory-breakdown').replaceChildren();
  DEVICES.forEach(device => {
    const included = members.includes(device), bar = document.createElement('span');
    bar.style.cssText = `width:${included ? device.memory / total * 100 : 0}%;background:${device.color}`;
    $('memory-track').append(bar);
    const row = document.createElement('div'), term = document.createElement('dt'), value = document.createElement('dd');
    const key = document.createElement('i'); key.className = 'key'; key.style.background = device.color;
    term.append(key, document.createTextNode(device.name));
    value.textContent = included ? `${device.memory.toFixed(1)} GiB` : device.id === 'mac' && joined ? 'Calculating…' : 'Not connected';
    row.append(term, value); $('memory-breakdown').append(row);
  });
  const resource = {off:'Start a session to add memory.',waiting:'Waiting for the MacBook to join.',reading:'Reading the MacBook’s specifications…',calculating:'Calculating resources…',splitting:'Assigning work to all three computers.',loading:'Preparing the model across the pool.',ready:'Three computers. One model.'};
  setText('resource-state', resource[phase]);
  const model = {off:'Waiting for the pool',waiting:'Waiting for your friend',reading:'Reading specifications…',calculating:'Calculating the split…',splitting:'Assigning layers…',loading:'Loading model…',ready:'Ready'};
  setText('model-state', model[phase]); $('model-state').classList.toggle('ready', ready);
  $('prompt').disabled = !ready || !connected || chatBusy;
  $('ask').disabled = !ready || !connected || chatBusy || !$('prompt').value.trim();
  document.querySelectorAll('[data-prompt]').forEach(button => button.disabled = !ready || !connected || chatBusy);
  setText('chat-state', chatBusy ? 'Responding…' : ready ? 'Ready to chat' : 'Available after loading');
}
async function allocate() {
  const run = epoch;
  phase = 'reading'; render();
  if (!await wait(2500, run)) return;
  phase = 'calculating'; render();
  if (!await wait(3500, run)) return;
  phase = 'splitting'; render();
  $('host-layers').textContent = 'GX10 · 56 layers'; $('friend-layers').textContent = 'GeForce RTX · 3  /  MacBook Air · 2';
  $('layer-legend').classList.add('allocated');
  const sequence = DEVICES.flatMap(device => Array(device.layers).fill(device));
  if (reducedMotion) {
    if (!await wait(4200, run)) return;
    sequence.forEach((device, index) => layerNodes[index].style.background = device.color);
  } else {
    for (let index = 0; index < sequence.length; index++) {
      if (!await wait(69, run)) return;
      layerNodes[index].style.background = sequence[index].color;
    }
  }
  $('layers').setAttribute('aria-label', 'Simulated split: GX10 56 layers, GeForce RTX 3 layers, MacBook Air 2 layers.');
  phase = 'loading'; render();
  if (!await wait(4500, run)) return;
  phase = 'ready'; render();
}
async function copy(text) {
  try {await navigator.clipboard.writeText(text);}
  catch (_) {
    const field = document.createElement('textarea'); field.value = text;
    field.style.cssText = 'position:fixed;left:-9999px;top:0'; document.body.append(field); field.select();
    const ok = document.execCommand('copy'); field.remove(); $('copy-link').focus();
    if (!ok) throw new Error('Clipboard unavailable. Share the code and open /join on this server.');
  }
}
$('hosting').onclick = async () => {
  busyToggle = true; render();
  try {adopt(await api('/api/hosting', {enabled:!state.hosting}));}
  catch (error) {showError(error);}
  finally {busyToggle = false; render();}
};
$('copy-link').onclick = async () => {
  if (!state.hosting) return;
  const run = epoch; copyError = ''; render();
  try {
    await copy(`${state.share_url}\nSession code: ${state.code}`);
    if (run !== epoch) return;
    copyNotice = 'Copied. Send it to your friend.'; render();
  } catch (error) {if (run === epoch) {copyError = error.message; render();}}
};
$('code-input').oninput = () => {$('code-input').value = $('code-input').value.toUpperCase().replace(/[^A-Z0-9]/g, ''); joinError = ''; render();};
$('join-form').onsubmit = async event => {
  event.preventDefault(); if (joinBusy || state.joined || !state.hosting) return;
  joinBusy = true; joinError = ''; render();
  try {adopt(await api('/api/join', {code:$('code-input').value.trim()}));}
  catch (error) {joinError = error.message;}
  finally {joinBusy = false; render();}
};
function answerFor(prompt) {
  if (/each|which|where|layer|computer do/i.test(prompt)) return 'The GX10 holds most of the model. The GeForce RTX laptop and MacBook Air each hold a smaller group of layers. Each token passes through those groups in order to produce one answer.';
  if (/memory|resource|fit|size/i.test(prompt)) return 'Sharing a model lets you use memory across several computers. Each machine stores its assigned layers, and the pool coordinates the work. More memory lets you fit larger models, but network transfers can make generation slower.';
  if (/why|split|share|together|pool/i.test(prompt)) return 'A large model can need more memory than one computer has. Splitting its layers across the GX10, your RTX laptop and a friend’s MacBook lets the computers work on the same answer together.';
  return 'This is a prepared demo response. In a live session, your question would run through the model layers across the pool, and the generated answer would stream back here.';
}
$('chat-form').onsubmit = async event => {
  event.preventDefault();
  if (phase !== 'ready' || !state.hosting || !connected || chatBusy || !$('prompt').value.trim()) return;
  const run = epoch, reply = answerFor($('prompt').value);
  chatBusy = true; $('response').hidden = false; $('response').setAttribute('aria-busy', 'true'); $('answer').textContent = ''; render();
  if (!await wait(1400, run)) return;
  for (const word of reply.match(/\S+\s*/g)) {
    if (!await wait(reducedMotion ? 0 : 75, run)) return;
    $('answer').textContent += word;
  }
  $('response').setAttribute('aria-busy', 'false'); chatBusy = false; render();
};
$('prompt').oninput = render;
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => {
  $('prompt').value = button.dataset.prompt; render(); $('prompt').focus();
});
function showError(error) {
  connected = false;
  setText('connection-error', !guest && !hostToken ? 'Open the private host launch link to control this session. Friends can use the join page.' : error.message);
  $('connection-error').hidden = false; render();
}
async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {adopt(await api(guest ? '/api/join-info' : '/api/state'));}
  catch (error) {showError(error);}
  finally {pollBusy = false;}
}
$('host-view').hidden = guest; $('join-view').hidden = !guest;
if (guest) {$('reset-note').hidden = true; document.title = 'Join the pool · Hypercompute';}
render(); poll(); setInterval(poll, 1000);
