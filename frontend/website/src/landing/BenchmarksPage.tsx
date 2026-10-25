import { BenchmarksContent } from './BenchmarksContent';
import { SiteHeader } from './SiteHeader';
import { SiteFooter } from './SiteFooter';
import './landing.css';

export function BenchmarksPage() {
  return (
    <div id="hc-home">
      <div className="hc-page hc-benchmarks-page">
        <a className="skip-link" href="#main-content">Skip to content</a>
        <SiteHeader page="benchmarks" />
        <BenchmarksContent />
        <SiteFooter page="benchmarks" />
      </div>
    </div>
  );
}
