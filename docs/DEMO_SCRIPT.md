# Two-minute demo script

Only say what the evidence on screen supports. Numbers below come from `docs/RESULTS.md`; replace them with
the run you just did.

**Setup before the demo** (not on stage): host and friend machines on the same hotspot or router, engine
installed on both, models downloaded on the host, dashboard built, one warm run done so nothing downloads live.

1. **(0:00) The ask.** "Hypercompute lets friends plug their computers into one session and run a local model
   together. Everything stays on this network."
2. **(0:15) Create a session.** On the host dashboard click **Create compute session**. Point at the one-time
   code and its expiry. "Codes are single-use and expire; each friend gets their own."
3. **(0:30) Friend joins.** On the friend's laptop run the shown `uv run hypercompute join ...` command. Show the
   friend terminal: detected hardware, the soft budget, the confirmation prompt. On the host, their card appears,
   goes `verifying` then `eligible` once the host completed a real llama.cpp RPC handshake with their worker.
4. **(0:50) Plan.** Pick the model, tick the friend, **Run together**, **Check plan**. Read the plan honestly:
   which layers go where, estimated memory vs budget, and the warning if the model would fit on the host alone
   ("this demonstrates participation; it is not faster").
5. **(1:05) Run.** Click **Run together**. The stepper goes planned, reserving, preparing, loading, warming, ready.
   Open **Technical details**: llama.cpp's own log shows the friend's `RPC0[ip:port]` model buffer and layer count.
6. **(1:20) Chat.** Ask a short question; tokens stream. Show TTFT and llama.cpp's decode rate.
7. **(1:35) Measured comparison.** Show the benchmark table: same model, this machine only vs together. Say what
   it shows: on the idle G14 the 7B decodes at ~81 tok/s alone and ~26 tok/s with 9 layers on a CPU peer
   (`final-*` rows in docs/RESULTS.md).
   "Adding a peer buys capacity, not speed. For models that do not fit one GPU, that capacity is the point."
8. **(1:50) Stop sharing.** The friend clicks **Stop sharing** (or Ctrl+C). The host shows the deployment failing
   cleanly with the reason, processes stopped, memory released. "Friends always keep the off switch."

Optional, only after it is verified on the day: unplug the internet uplink (LAN still up) and send another
request to show nothing depends on the cloud.

Claims to avoid: "the model cannot run on any single machine" (unless measured on every machine present),
"faster with more friends" (not what the measurements show), "secure over the internet" (RPC is unauthenticated;
trusted LAN only).
