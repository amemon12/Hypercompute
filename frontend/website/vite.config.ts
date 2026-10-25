import { defineConfig, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';
function localProxy(port: number, service: string): ProxyOptions {
  return {
    target: `http://127.0.0.1:${port}`, changeOrigin: true,
    configure(proxy) {
      proxy.on('error', (_error, _request, response) => {
        if ('writeHead' in response && !response.headersSent) response.writeHead(502, { 'Content-Type': 'application/json' }).end(JSON.stringify({
          code: `${service}_unreachable`, detail: `${service === 'gateway' ? 'Hypercompute host' : 'Local connector'} is not running on this computer.`,
          fix: service === 'gateway' ? 'Run uv run hypercompute host, then reconnect.' : 'Start the Hypercompute connector on this computer, then reconnect.',
        }));
      });
    },
  };
}
const proxy = { '/api': localProxy(8000, 'gateway'), '/v1': localProxy(8000, 'gateway'), '/local-api': localProxy(8001, 'connector') };
export default defineConfig({ plugins: [react()], server: { host: '127.0.0.1', port: 3000, strictPort: true, proxy, fs: { allow: ['..'] } }, preview: { host: '127.0.0.1', port: 3000, strictPort: true, proxy } });
