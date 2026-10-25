import { defineConfig, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

// The gateway (FastAPI) listens on 127.0.0.1:8000. Owner endpoints check that the Host header is
// 127.0.0.1:<port> or localhost:<port>, so the dev proxy must rewrite Host (changeOrigin).
const GATEWAY = 'http://127.0.0.1:8000';

// When the gateway is down, answer like the gateway would ({detail, code, fix}) instead of Vite's
// bare 500, so the dashboard can say what is wrong.
const UNREACHABLE = JSON.stringify({
  detail: `Cannot reach the Hypercompute gateway at ${GATEWAY}`,
  code: 'gateway_unreachable',
  fix: 'Start the gateway on this computer, then reload this page.',
});

const gatewayProxy: ProxyOptions = {
  target: GATEWAY,
  changeOrigin: true,
  configure: (proxy) => {
    proxy.on('error', (_err, _req, res) => {
      if ('writeHead' in res && !res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'application/json' }).end(UNREACHABLE);
      }
    });
  },
};

export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    proxy: {
      '/api': gatewayProxy,
      '/v1': gatewayProxy,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
