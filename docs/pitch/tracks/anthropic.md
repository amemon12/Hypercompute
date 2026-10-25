# [Anthropic] Claude for Public Benefit, Cal Hacks 13.0 (1 winner. $5,000 Claude credits, 1 year of Claude Max 20x and office hours. Devpost lists it at $4,998)

Fit 5/10, prize about $4998, effort about 6 h (estimate).

## Integration

VERDICT. The fit is moderate, not strong. Hypercompute's core (pooled local llama.cpp inference) works without Claude. The only role where Claude does real work without breaking the product's privacy premise is a "Pool Doctor". When a deployment fails or a peer drops, Claude turns the gateway's own telemetry into a grounded diagnosis plus one allowlisted fix, and the owner clicks it. Claude only sees redacted machine telemetry. It never sees prompts, outputs or data, and it never executes anything.

NAMED COMMUNITY (narrow it further after talking to people tonight). Berkeley undergrad research teams, such as URAP projects or student ML-club project teams. They need 100B+ open-weight models for evals, or for data their IRB or data-use agreement keeps off hosted APIs. They have only their members' laptops and no GPU budget, and nobody on the team is a distributed-systems person. The lived problem is the team's own 4-day failure log, already in the repo. Examples: Windows reserving ports 50000-50059 (config.py), apartment Wi-Fi client isolation forcing the relay, a peer on the relay at 47-84 ms RTT, a peer catalog missing a model causing a 4 MiB/s host push (CHECKPOINT-2026-10-24), and the planner predicting 1.77x but measuring 0.83x (DEVPOST.md). To meet the "talked to someone" bar, get 2-3 Berkeley students at the venue tonight and quote one in the Devpost.

WHY CLAUDE AND NOT RULES. The repo already has about 110 deterministic fix= hints (Diagnostic.fix, VerificationResult.fix, the DeploymentError fix field). Those stay first. Claude adds two things the rules cannot. (a) Correlation across nodes and time. For example, a stale heartbeat, relay mode, rising lost_packets and a seed error on one Mac together mean "the link, not the Mac". (b) The long tail of raw llama.cpp and engine error strings that no rule maps.

BACKEND (new code only, about 350 lines plus tests)
1. New backend/hypercompute/services/assist_snapshot.py. Pure functions, no network. build_snapshot(coord, deployment_id) collects:
   - from coord.nodes.views(): display_name, OS, CPU, memory_pools, devices, disk_free, state, state_reason, heartbeat_age_s, worker_state, verification.detail and fix, inventory.diagnostics, contribution
   - the deployment: state, error, cleanup, progress, history, plan stages and predictions with their heuristic flag, participation_detail
   - app.state.peer_service.status() paths: mode relay or direct, rtt_ms, lost_packets, keyed by node id
   - /api/peers seed progress and errors
   - the last 60 EventBus events
   - RunRecord summaries (status, error, ttft_ms, timings), with output_text and messages_sha256 removed

   Redaction removes or replaces:
   - hostname, replaced by node id or display_name
   - network[].ipv4, worker.host, transport_peer_id, endpoint_id, peer_id and remote_id
   - IPv4 and IPv6 addresses, home-directory paths, and invite or owner-token shapes, all caught by regex

   The snapshot is capped at about 40 KB.
2. New backend/hypercompute/services/assist.py. AssistService.diagnose(snapshot, question):
   - Calls anthropic.AsyncAnthropic(). The key comes from ANTHROPIC_API_KEY on the host only. The model comes from HYPERCOMPUTE_ASSIST_MODEL, default claude-opus-5-5.
   - Request settings: output_config effort medium plus a json_schema format for AssistDiagnosis. No temperature (it is now a 400). No forced tool_choice (a 400 on Opus 5.5).
   - Enable fallbacks "default" with the server-side-fallback-2026-07-01 beta on client.beta.messages.
   - System prompt: frozen instructions plus docs/RUNBOOK.md (with its 6 IP and path strings scrubbed), under top-level cache_control so repeat clicks hit the cache.
   - User message: the snapshot, labeled as untrusted data, plus the owner's optional question (500 characters max).
   - Handle a refusal or max_tokens stop_reason, and catch the typed SDK errors. Map them to DeploymentError 502 or 503 with a fix hint.
3. Server-side validation, which is what makes the integration trustworthy:
   - Each evidence item must carry a JSON pointer into the snapshot plus the value it cites. Drop items whose pointer does not resolve or whose value does not match.
   - Actions must come from an enum mapped to existing endpoints: verify_node (POST /api/nodes/{id}/verify), propose_plan (its args validated as PlanRequest, then POST /api/plans), seed_peer (POST /api/peers/{id}/seed) and stop_deployment (DELETE /api/deployments/{id}). Every id must exist in the snapshot.
   - Steps for other machines, such as a firewall rule on a friend's Mac, are shown as text for a named machine and never executed.
   - If no grounded evidence survives, return the deterministic fix hints instead.
4. schemas.py: add AssistRequest, AssistEvidence, AssistAction, AssistDiagnosis, and AssistResponse (diagnosis, sent_snapshot, model, usage).
5. app.py: add POST /api/assist/diagnose with OwnerWrite (loopback plus owner token, because it spends money).
6. coordinator.py: add the field assist: AssistService | None.
7. pyproject.toml: add an optional extra, assist = ["anthropic"], imported lazily so friends' peer installs never pull it. The 1.x SDK uses httpx2 next to the repo's httpx. Pin the version after checking PyPI.

FRONTEND
- api/client.ts: add diagnose().
- New components/AssistPanel.tsx. It appears under the failed-deployment note in DeploymentPanel.tsx (lines 92-98) and on stale or ineligible PeerCards. It shows:
  - the summary, likely cause and confidence
  - evidence chips naming the cited field
  - action buttons that reuse existing calls, with propose_plan opening the existing PlanView so the owner clicks Deploy
  - a "What Claude saw" disclosure showing sent_snapshot verbatim

TESTS (offline, a fake client injected through a Protocol, run with uv run pytest)
- Redaction never leaks an IP, hostname, home path, invite or output_text.
- Ungrounded evidence is dropped.
- An action with an unknown node is rejected.
- A missing key gives a 503 with a fix.
- A non-loopback request or a missing token gets a 401.
- The panel renders under vitest.

DATA FLOW. Owner clicks. The gateway builds and redacts the snapshot locally. One Claude call. The gateway validates the diagnosis. The owner clicks one allowlisted action, which calls an existing endpoint. Prompts, outputs and weights never leave the pool.

## Demo moment

At the table, a 120B model is serving on the G14 plus 3 MacBook Airs.
1. A teammate kills mac-b's worker, or moves it onto a phone hotspot. The node goes stale and the deployment shows a raw failure string.
2. Click "Ask Claude why". In about 15 s the panel says mac-b dropped, and cites its heartbeat age and its link switching to relay at hundreds of ms RTT, each as a clickable field.
3. It says the 120B still fits on the G14, mac-a and mac-azim, and offers a "Plan without mac-b" button. It also gives a text step for mac-b: rejoin with a new code once back on venue Wi-Fi.
4. Click the button. PlanView shows the new layout. Click Deploy, and chat answers again.
5. Open "What Claude saw" to show the judges the redacted JSON: no IPs, no hostnames, no prompts.
Close with the pitch line: Claude is the sysadmin a student research club does not have, and it never touches their data. Keep a pre-recorded clip as backup in case the venue Wi-Fi fails.

## Accounts and keys (the user creates these)

The user does all of this; Claude never signs up or handles keys.
- Create an Anthropic Console account (platform.claude.com) and an API key. First ask at the Anthropic table or Discord whether there are event credit codes. The prize page lists none for hackers.
- Export ANTHROPIC_API_KEY only in the gateway host's shell environment (the G14, or the GX10 if it hosts). Never put it in the repo, configs, the dashboard bundle or a peer.
- Optional: set HYPERCOMPUTE_ASSIST_MODEL to override the default claude-opus-5-5.
- Cost: at $4 in and $20 out per million tokens, with the runbook cached, each diagnosis is roughly $0.05 to $0.20, so $10 covers the event.
- Peers need nothing new.

## Risks

1. Weak to moderate fit. Claude is an ops layer, and the product works without it. The rubric rewards Claude doing something that matters to the community's problem, so judges may read this as bolted on. The only answer is a demo where recovery visibly depends on it. The no-data-leaves-the-pool story has to be shown, not claimed.
2. Community not validated yet. "Berkeley student research teams" is close to the "Students" breadth the rubric calls too broad, and no conversations are on record yet. Narrow it to one real group with a quote, or the first criterion is weak.
3. Wrong diagnosis. Claude could confidently advise something harmful, such as turning off a friend's firewall. The repo has already shown that predictions mislead: the planner said 1.77x and measured 0.83x. Mitigations:
   - Server-side grounding checks.
   - The action allowlist and id validation.
   - No execution, ever.
   - Plan predictions labeled heuristic.
   - Measured run numbers only.
4. Prompt injection through peer-reported strings: display names, error text, Hugging Face filenames. Treat them as data, give Claude no side-effect tools, and validate every action.
5. Privacy regression in a privacy-first tool. A missed field could leak a LAN IP, hostname, home path or the live invite (the host log holds one). Mitigations: redaction tests, the verbatim disclosure, and keeping the feature opt-in and off without a key.
6. Safety-classifier false positives on network or firewall troubleshooting, under Opus 5.5's cyber category. Handle the refusal stop_reason, enable the default server-side fallbacks, and degrade to the deterministic fix hints.
7. Latency of 10 to 30 s and venue Wi-Fi at judging. Warm the prompt cache before 10:30, stage the failure by killing the worker rather than relying on Wi-Fi, and keep a recorded clip.
8. Time. About 15 h remain, overnight benchmarks are running, and the Devpost is due at 9:00 AM PDT. 6 h of build competes with the pitch, the deck and the video.
9. Eligibility, which applies to every track. The Cal Hacks FAQ bans projects worked on before the event, and Hypercompute's commits date from Oct 20 to 24. The Pool Doctor would be new code, but the base is pre-existing. Confirm with the organizers or disclose, because a disqualification voids this prize too.

## Sources

- https://hive.hackberkeley.org/prizes (Anthropic criteria: named community, real problem, it works, thoughtful use of Claude)
- https://cal-hacks-13-0.devpost.com/ (prize listing: $4,998 value, $5000 Claude credits + 1 year Claude Max 20x + Office Hours, 1 winner)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\app.py (owner_read/owner_write loopback+token guards, routes /api/nodes/{id}/verify, /api/peers/{id}/seed, /api/plans, /api/deployments/{id}/replace, /api/replan/trigger, /api/network)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\schemas.py (Diagnostic.fix, VerificationResult.fix, NodeView, HostInventory with hostname/ipv4, Deployment.error/cleanup, RunRecord.output_text, PlanRequest)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\coordinator.py (service graph where an AssistService would attach)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\transport\host.py and transport\iroh.py (path_info: relay/direct mode, rtt_ms, lost_packets)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\events.py (EventBus, 500-event ring)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\config.py (Windows reserved ports note)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\frontend\src\components\DeploymentPanel.tsx (failed-deployment note, lines 92-98)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\RUNBOOK.md (troubleshooting table, 11.7 KB, 6 IP/path strings to scrub)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\pitch\DEVPOST.md (challenges: Wi-Fi isolation, relay, 1.77x predicted vs 0.83x measured)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\research\CHECKPOINT-2026-10-24.md (catalog-mismatch 4 MiB/s push, relay peers, live invite in host log)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\pitch\ultraplan-scouts.json (prior-work rule quote, prize list)
- claude-api skill (bundled, cached 2026-10-06): claude-opus-5-5 at $4/$20 per MTok, effort default medium, forced tool_choice 400, sampling params 400 after 2026-10-21, output_config.format structured outputs, fallbacks default with server-side-fallback-2026-07-01, anthropic 1.x on httpx2
