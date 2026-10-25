"""Friend-side local control: a loopback-only status page with a 'Stop sharing' button.

The contributor always keeps local stop control, even when the host is unreachable.
"""

from __future__ import annotations

import asyncio
import hmac
import html
import json
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

LOOPBACK = {"127.0.0.1", "::1"}
TOKEN_HEADER = "X-Hypercompute-Agent"  # noqa: S105 - a header name, not a secret
# the page embeds the token and a Stop sharing button: another site must not frame it
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Hypercompute sharing</title><style>
:root{{color-scheme:dark}}body{{font:15px/1.5 system-ui,sans-serif;background:#111315;color:#e6e6e6;margin:0;padding:24px}}
main{{max-width:640px;margin:auto}}h1{{font-size:20px}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px}}
dt{{color:#9aa0a6}}button{{background:#e5484d;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;
cursor:pointer}}button.alt{{background:#2b2f33}}.muted{{color:#9aa0a6}}</style></head><body><main>
<h1>This computer is sharing compute with {host}</h1>
<dl id="s"></dl><p><button id="pause" class="alt">Pause</button> <button id="resume" class="alt">Resume</button>
<button id="stop">Stop sharing</button></p>
<p class="muted">Pause stops the local worker but stays in the session; the host stops using this machine until you
resume. Stop sharing ends the worker and tells the host you left.</p>
<script>
const T={token};
async function load(){{const r=await fetch('/status');const s=await r.json();const d=document.getElementById('s');
d.innerHTML='';for(const [k,v] of Object.entries(s)){{const dt=document.createElement('dt');dt.textContent=k;
const dd=document.createElement('dd');dd.textContent=typeof v==='object'?JSON.stringify(v):String(v);d.append(dt,dd);}}}}
async function post(p){{await fetch(p,{{method:'POST',headers:{{'{header}':T}}}});setTimeout(load,800);}}
document.getElementById('pause').onclick=()=>post('/pause');document.getElementById('resume').onclick=()=>post('/resume');
document.getElementById('stop').onclick=async()=>{{await post('/leave');document.getElementById('stop').disabled=true;}};
load();setInterval(load,3000);
</script></main></body></html>"""


def create_control_app(token: str, status: Callable[[], dict], request_leave: Callable[[], None],
                       host_label: str, pause: Callable[[], str] | None = None,
                       resume: Callable[[], str] | None = None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def local_only(request: Request) -> bool:
        client = request.client.host if request.client else ""
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
        return client in LOOPBACK and host in ("127.0.0.1", "localhost")

    @app.get("/", response_class=HTMLResponse)
    async def page(request: Request) -> HTMLResponse:
        if not local_only(request):
            return HTMLResponse("forbidden", status_code=403)
        return HTMLResponse(PAGE.format(host=html.escape(host_label), token=json.dumps(token), header=TOKEN_HEADER))

    @app.get("/status")
    async def get_status(request: Request) -> JSONResponse:
        if not local_only(request):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        return JSONResponse(status())

    def authorized(request: Request, given: str | None) -> bool:
        return local_only(request) and bool(given) and hmac.compare_digest(given or "", token)

    @app.post("/leave")
    async def leave(request: Request, x_hypercompute_agent: str | None = Header(default=None)) -> JSONResponse:
        if not authorized(request, x_hypercompute_agent):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        request_leave()
        return JSONResponse({"ok": True})

    @app.post("/pause")
    async def do_pause(request: Request, x_hypercompute_agent: str | None = Header(default=None)) -> JSONResponse:
        if not authorized(request, x_hypercompute_agent) or pause is None:
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        return JSONResponse({"ok": True, "detail": await asyncio.to_thread(pause)})

    @app.post("/resume")
    async def do_resume(request: Request, x_hypercompute_agent: str | None = Header(default=None)) -> JSONResponse:
        if not authorized(request, x_hypercompute_agent) or resume is None:
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        try:
            return JSONResponse({"ok": True, "detail": await asyncio.to_thread(resume)})
        except RuntimeError as exc:  # worker failed to restart; the friend sees why
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=500)

    return app
