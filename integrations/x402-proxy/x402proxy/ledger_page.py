"""The /ledger page: what each friend's machine earned, refreshed every 3 seconds from /api/ledger.

Every value from the ledger (machine names come from friends' own machines) is set with textContent, never as HTML.
"""

LEDGER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pool Ledger</title>
<style>
:root { --bg:#f7f7f5; --card:#ffffff; --ink:#16181d; --muted:#626873; --line:#e3e4e8; --accent:#1652f0;
        --good:#11845b; --warn:#b4540a; --bar:#dce5fd; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --card:#171a21; --ink:#eceef2; --muted:#9aa1ad; --line:#2a2f3a; --accent:#6f9bff;
          --good:#3ccf91; --warn:#f0a35e; --bar:#22304f; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 ui-sans-serif, system-ui, -apple-system,
       "Segoe UI", sans-serif; }
main { max-width:1040px; margin:0 auto; padding:28px 16px 48px; }
header { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px 16px; margin-bottom:6px; }
h1 { font-size:22px; margin:0; letter-spacing:-0.01em; }
.tag { font-size:12px; border:1px solid var(--warn); color:var(--warn); border-radius:999px; padding:1px 9px; }
.sub { color:var(--muted); margin:0 0 22px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:22px; }
.stat, .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.stat b { display:block; font-size:22px; font-variant-numeric:tabular-nums; }
.stat span { color:var(--muted); font-size:13px; }
h2 { font-size:15px; margin:26px 0 10px; }
.machines { display:grid; gap:10px; }
.machine { display:grid; grid-template-columns:minmax(120px, 1.2fr) 3fr auto; gap:12px; align-items:center; }
.name { font-weight:600; overflow-wrap:anywhere; }
.name small { display:block; font-weight:400; color:var(--muted); font-size:12px; }
.bar { height:10px; background:var(--bar); border-radius:999px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--accent); border-radius:999px; }
.earned { font-variant-numeric:tabular-nums; font-weight:600; text-align:right; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th, td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:500; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.status-delivered { color:var(--good); } .status-interrupted { color:var(--warn); } .status-settled { color:var(--muted); }
a { color:var(--accent); }
.chips span { display:inline-block; border:1px solid var(--line); border-radius:6px; padding:0 6px; margin:0 4px 4px 0;
              font-size:12px; white-space:nowrap; }
.scroll { overflow-x:auto; }
.empty { color:var(--muted); padding:12px 0; }
@media (max-width: 560px) { .machine { grid-template-columns:1fr auto; } .machine .bar { grid-column:1 / -1; } }
</style>
</head>
<body>
<main>
  <header><h1>Hypercompute pool ledger</h1><span class="tag">Base Sepolia testnet, test USDC only</span></header>
  <p class="sub">Each paid chat is split by the layers each machine served in the deployment plan. The host keeps a
    fee for running the gateway. Updates every 3 seconds.</p>
  <section class="stats">
    <div class="stat"><b id="s-payments">0</b><span>paid chats</span></div>
    <div class="stat"><b id="s-paid">0.000000</b><span>USDC paid by agents</span></div>
    <div class="stat"><b id="s-refunds">0.000000</b><span>USDC refunds owed (cut streams)</span></div>
  </section>
  <h2>Earned per machine</h2>
  <section class="card machines" id="machines"><div class="empty">No paid chats yet.</div></section>
  <h2>Recent payments</h2>
  <section class="card scroll">
    <table>
      <thead><tr><th>When</th><th>Paid</th><th>Model</th><th>Status</th><th>Tokens</th><th>Split</th><th>Tx</th></tr></thead>
      <tbody id="recent"><tr><td colspan="7" class="empty">No paid chats yet.</td></tr></tbody>
    </table>
  </section>
  <h2>Payouts to friends (on chain)</h2>
  <section class="card scroll">
    <table>
      <thead><tr><th>When</th><th>Machine</th><th>Amount</th><th>Status</th><th>To</th><th>Tx</th></tr></thead>
      <tbody id="payouts"><tr><td colspan="6" class="empty">No payouts yet. Run payout.py to settle up.</td></tr></tbody>
    </table>
  </section>
</main>
<script>
function el(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls; return e; }
function short(s) { return s && s.length > 14 ? s.slice(0, 8) + "..." + s.slice(-4) : (s || ""); }
function renderMachines(totals) {
  const box = document.getElementById("machines"); box.replaceChildren();
  if (!totals.length) { box.append(el("div", "No paid chats yet.", "empty")); return; }
  const max = Math.max(...totals.map(t => Number(t.earned_atomic)), 1);
  for (const t of totals) {
    const row = el("div", undefined, "machine");
    const name = el("div", t.display_name, "name");
    const paid = t.is_host ? "receives every payment directly"
      : (Number(t.paid_out_atomic) > 0 ? t.paid_out_usdc + " paid out on chain" : "not paid out yet");
    name.append(el("small", (t.is_host ? "host" : "peer") + " / " +
      (t.payout_address ? short(t.payout_address) : "no payout address yet") + " / " + paid));
    const bar = el("div", undefined, "bar"); const fill = el("i");
    fill.style.width = (100 * Number(t.earned_atomic) / max).toFixed(1) + "%"; bar.append(fill);
    row.append(name, bar, el("div", t.earned_usdc + " USDC", "earned")); box.append(row);
  }
}
function renderRecent(recent, explorer) {
  const body = document.getElementById("recent"); body.replaceChildren();
  if (!recent.length) { const tr = el("tr"); const td = el("td", "No paid chats yet.", "empty"); td.colSpan = 7;
    tr.append(td); body.append(tr); return; }
  for (const p of recent) {
    const tr = el("tr");
    tr.append(el("td", new Date(p.created_at).toLocaleTimeString(), "num"));
    tr.append(el("td", p.amount_usdc, "num"));
    tr.append(el("td", p.model_id));
    const st = el("td", p.status, "status-" + p.status);
    if (p.status === "interrupted") st.title = (p.note || "") + " / refund owed " + p.refund_owed_usdc + " USDC";
    tr.append(st);
    tr.append(el("td", (p.completion_tokens ?? "-") + " / " + p.max_tokens, "num"));
    const chips = el("td", undefined, "chips");
    for (const s of p.shares) chips.append(el("span", s.display_name + " " + s.layers + " = " + s.amount_usdc));
    tr.append(chips, txCell(p.tx, explorer)); body.append(tr);
  }
}
function txCell(tx, explorer) {
  const td = el("td", undefined, "num");
  if (tx && explorer && /^0x[0-9a-fA-F]{64}$/.test(tx)) { const a = el("a", short(tx)); a.href = explorer + tx;
    a.target = "_blank"; a.rel = "noopener"; td.append(a); } else td.textContent = short(tx) || "-";
  return td;
}
function renderPayouts(payouts, explorer) {
  const body = document.getElementById("payouts"); body.replaceChildren();
  if (!payouts.length) { const tr = el("tr"); const td = el("td", "No payouts yet. Run payout.py to settle up.", "empty");
    td.colSpan = 6; tr.append(td); body.append(tr); return; }
  for (const p of payouts) {
    const tr = el("tr");
    tr.append(el("td", new Date(p.created_at).toLocaleTimeString(), "num"), el("td", p.display_name),
              el("td", p.amount_usdc, "num"), el("td", p.status), el("td", short(p.address), "num"),
              txCell(p.tx, explorer));
    body.append(tr);
  }
}
async function refresh() {
  try {
    const r = await fetch("/api/ledger", {cache: "no-store"}); if (!r.ok) return; const d = await r.json();
    document.getElementById("s-payments").textContent = d.summary.payments;
    document.getElementById("s-paid").textContent = d.summary.paid_usdc;
    document.getElementById("s-refunds").textContent = d.summary.refunds_owed_usdc;
    renderMachines(d.totals); renderRecent(d.recent, d.explorer_tx_url); renderPayouts(d.payouts, d.explorer_tx_url);
  } catch (e) { console.warn("ledger refresh failed", e); }
}
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>
"""
