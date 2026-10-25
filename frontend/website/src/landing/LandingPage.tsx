import { ExecutionDiagram } from './ExecutionDiagram';
import { SiteHeader } from './SiteHeader';
import { SiteFooter } from './SiteFooter';
import { SessionSection } from './SessionSection';
import './landing.css';

export function LandingPage() {
  return (
    <div id="hc-home">
      <div className="hc-page">
        <a className="skip-link" href="#main-content">Skip to content</a>
        <SiteHeader />
        <div className="hc-content">
          <main id="main-content" tabIndex={-1}>
            <section className="hc-hero hc-panel" id="overview" aria-labelledby="home-title">
              <div className="hc-hero-copy">
                <h1 className="hc-title" id="home-title">Run big models together.</h1>
                <p className="hc-lede">Start a session on one computer and connect the others. Hypercompute splits the model’s layers across their GPUs over your local network.</p>
              </div>
              <figure className="hc-hardware-study" aria-label="GPU illustration">
                <div className="hc-hardware-canvas">
                  <img
                    className="hc-board-image"
                    src="/brand/gpu-retro-paper.png"
                    alt="Retro print of a conceptual hybrid GPU, with copper details and a green ink shape on textured paper"
                    width="2230"
                    height="1264"
                    loading="eager"
                    decoding="async"
                  />
                </div>
              </figure>
            </section>

            <SessionSection />

            <section className="hc-compute-study hc-panel" id="hardware" aria-label="An example with three computers">
              <ExecutionDiagram />
            </section>
          </main>
        </div>

        <SiteFooter />
      </div>
    </div>
  );
}
