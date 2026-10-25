import { useEffect, useState } from 'react';
import './scroll-brand.css';

export function ScrollBrand({ href = '#overview' }: { href?: string }) {
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      // The small dead zone keeps touchpad movement from toggling the mark.
      setCompact(previous => window.scrollY > (previous ? 16 : 64));
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('pageshow', onScroll);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('pageshow', onScroll);
    };
  }, []);

  return (
    <a className="hc-logo" href={href} aria-label="Hypercompute home" data-compact={compact}>
      <span className="hc-brand-word" aria-hidden="true">
        <span className="hc-brand-prefix"><span>hyperc</span></span>
        <span className="hc-brand-core">
          <span className="hc-brand-o">o</span>
          <span className="hc-brand-die" />
          <svg className="hc-brand-pins" viewBox="0 0 40 44" focusable="false">
            <path d="M14 10V5 M20 10V5 M26 10V5 M14 34V39 M20 34V39 M26 34V39 M8 16H3 M8 22H3 M8 28H3 M32 16H37 M32 22H37 M32 28H37" />
          </svg>
        </span>
        <span className="hc-brand-suffix"><span>mpute</span></span>
      </span>
    </a>
  );
}
