import { useEffect } from 'react';
import { BenchmarksPage } from './landing/BenchmarksPage';
import { LandingPage } from './landing/LandingPage';

export default function App() {
  const benchmarks = window.location.pathname.replace(/\/$/, '') === '/benchmarks';

  useEffect(() => {
    document.title = benchmarks
      ? 'Benchmarks | Hypercompute'
      : 'Hypercompute | Run big models together.';

    // Cross-page section links need their destination after React has mounted it.
    if (!window.location.hash) return;
    const target = document.getElementById(window.location.hash.slice(1));
    const frame = window.requestAnimationFrame(() => {
      target?.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [benchmarks]);

  return benchmarks ? <BenchmarksPage /> : <LandingPage />;
}
