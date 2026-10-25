#!/usr/bin/env bash
S=/c/Users/operator/AppData/Local/Temp/claude/C--/9e2c80d5-50ca-460d-8443-8db45359fcde/scratchpad/smoke120b
E=/c/Users/operator/.hypercompute/engines/llama-b11770-windows-x64-cuda13.4
M='C:\Users\operator\.hypercompute\iroh-session\models\gpt-oss-120b-mxfp4\gpt-oss-120b-MXFP4.gguf'
start=$(date +%s)
"$E/llama-server.exe" -m "$M" --load-mode mmap --no-repack -fa on --device CUDA0 -ngl all -ncmoe 33 --fit off -c 4096 -np 1 -t 12 --cache-ram 0 -ctxcp 2 --no-warmup --host 127.0.0.1 --port 8090 > "$S/server.log" 2>&1 &
pid=$!
echo "server pid $pid" > "$S/smoke.txt"
for i in $(seq 1 450); do
  if curl -s -m 2 http://127.0.0.1:8090/health | grep -q '"ok"'; then echo "ready after $(( $(date +%s) - start )) s" >> "$S/smoke.txt"; break; fi
  if ! kill -0 $pid 2>/dev/null; then echo "server exited during load" >> "$S/smoke.txt"; break; fi
  sleep 2
done
t0=$(date +%s)
curl -s -m 900 http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Explain mixture-of-experts in two sentences."}],"max_tokens":160,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"low"}}' > "$S/chat.json"
echo "chat wall $(( $(date +%s) - t0 )) s" >> "$S/smoke.txt"
taskkill //PID $pid //T //F > /dev/null 2>&1 || kill $pid
echo "stopped" >> "$S/smoke.txt"
