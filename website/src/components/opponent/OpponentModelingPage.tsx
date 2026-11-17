import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Button } from "@/components/Button";
import { BayesianLearner } from "@/components/opponent/BayesianLearner";
import { HiddenCardDiagram } from "@/components/opponent/HiddenCardDiagram";
import { M5ResultsPanel } from "@/components/opponent/M5ResultsPanel";
import { ChangeDetectionPanel } from "@/components/opponent/ChangeDetectionPanel";
import {
  bayesCopy,
  changeCopy,
  continueCopy,
  hiddenCopy,
  opponentIntro,
  opponentNav,
  resultsCopy,
} from "@/content/opponentModelingPage";
import { withBasePath } from "@/lib/paths";

export function OpponentModelingPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <SiteHeader
        brandHref={withBasePath("/")}
        navItems={[...opponentNav]}
        trailingHref={withBasePath("/")}
        trailingLabel="Home"
      />
      <main id="main" className="opponent-page">
        <section className="section cfr-hero" id="intro" aria-labelledby="opponent-title">
          <div className="shell cfr-hero__grid">
            <div>
              <p className="section-kicker">{opponentIntro.kicker}</p>
              <h1 id="opponent-title">{opponentIntro.headline}</h1>
              <p className="cfr-hero__lede">{opponentIntro.lede}</p>
              <p>{opponentIntro.plain}</p>
            </div>
            <aside className="cfr-hero__aside" aria-label="Exhibit map">
              <p className="mono research-banner__label">On this page</p>
              <ol className="cfr-toc">
                {opponentNav.map((item) => (
                  <li key={item.href}>
                    <a href={item.href}>{item.label}</a>
                  </li>
                ))}
              </ol>
            </aside>
          </div>
        </section>

        <section className="section" id="bayes" aria-labelledby="bayes-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Interactive</p>
              <h2 id="bayes-title">{bayesCopy.title}</h2>
              <p>{bayesCopy.summary}</p>
            </header>
            <BayesianLearner />
          </div>
        </section>

        <section className="section" id="hidden" aria-labelledby="hidden-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Limitation</p>
              <h2 id="hidden-title">{hiddenCopy.title}</h2>
              <p>{hiddenCopy.summary}</p>
            </header>
            <HiddenCardDiagram />
          </div>
        </section>

        <section className="section" id="results" aria-labelledby="results-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Measured</p>
              <h2 id="results-title">{resultsCopy.title}</h2>
              <p>{resultsCopy.summary}</p>
            </header>
            <M5ResultsPanel />
          </div>
        </section>

        <section className="section" id="change" aria-labelledby="change-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Non-stationary play</p>
              <h2 id="change-title">{changeCopy.title}</h2>
            </header>
            <ChangeDetectionPanel />
          </div>
        </section>

        <section className="section final-cta" aria-labelledby="continue-title">
          <div className="shell final-cta__panel">
            <h2 id="continue-title">{continueCopy.headline}</h2>
            <p>{continueCopy.body}</p>
            <div className="hero__actions">
              <Button href={withBasePath(continueCopy.primary.href)}>
                {continueCopy.primary.label}
              </Button>
              <Button href={withBasePath(continueCopy.secondary.href)} variant="secondary">
                {continueCopy.secondary.label}
              </Button>
              <Button href={continueCopy.tertiary.href} variant="ghost" external>
                {continueCopy.tertiary.label}
              </Button>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
