import { ScrollBrand } from './ScrollBrand';

export function SiteHeader({ page = 'home' }: { page?: 'home' | 'benchmarks' }) {
  const home = page === 'home' ? '' : '/';

  return (
    <header className="hc-nav">
      <div className="hc-nav-inner">
        <nav className="hc-section-nav hc-nav-left" aria-label="Overview and session">
          <a href={`${home}#overview`}>Overview</a>
          <a href={`${home}#how-it-works`}>Session</a>
        </nav>
        <ScrollBrand href={`${home}#overview`} />
        <nav className="hc-section-nav hc-nav-right" aria-label="Hardware and benchmarks">
          <a href={`${home}#hardware`}>Hardware</a>
          <a href="/benchmarks" aria-current={page === 'benchmarks' ? 'page' : undefined}>Benchmarks</a>
        </nav>
      </div>
    </header>
  );
}
