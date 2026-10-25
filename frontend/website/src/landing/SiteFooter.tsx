import { PortraitWordmark } from './PortraitWordmark';

export function SiteFooter({ page = 'home' }: { page?: 'home' | 'benchmarks' }) {
  const home = page === 'home' ? '' : '/';

  return (
    <footer className="hc-endpiece hc-panel" id="wordmark">
      <div className="hc-footer-info">
        <nav className="hc-footer-nav" aria-label="Footer">
          <a href={`${home}#overview`}>Overview</a>
          <a href={`${home}#how-it-works`}>Session</a>
          <a href={`${home}#hardware`}>Hardware</a>
          <a href="/benchmarks">Benchmarks</a>
        </nav>
        <div className="hc-footer-meta">
          <small>{new Date().getFullYear()} Hypercompute</small>
          <a href={page === 'home' ? '#overview' : '#benchmarks-top'}>Back to top</a>
        </div>
      </div>
      <PortraitWordmark />
    </footer>
  );
}
