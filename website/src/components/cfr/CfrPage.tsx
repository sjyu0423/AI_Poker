import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Button } from "@/components/Button";
import { RegretMatchingDemo } from "@/components/cfr/RegretMatchingDemo";
import { KuhnDiagram } from "@/components/cfr/KuhnDiagram";
import { TrainingTimeline } from "@/components/cfr/TrainingTimeline";
import {
  averageCopy,
  continueCopy,
  cfrIntro,
  cfrNav,
  kuhnCopy,
  regretMatchingCopy,
  trainingCopy,
  verificationCopy,
} from "@/content/cfrPage";
import { withBasePath } from "@/lib/paths";

export function CfrPage() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <SiteHeader
        brandHref={withBasePath("/")}
        navItems={[...cfrNav]}
        trailingHref={withBasePath("/")}
        trailingLabel="Home"
      />
      <main id="main" className="cfr-page">
        <section className="section cfr-hero" id="intro" aria-labelledby="cfr-title">
          <div className="shell cfr-hero__grid">
            <div>
              <p className="section-kicker">{cfrIntro.kicker}</p>
              <h1 id="cfr-title">{cfrIntro.headline}</h1>
              <p className="cfr-hero__lede">{cfrIntro.lede}</p>
              <blockquote className="cfr-quote">
                <p>{cfrIntro.centralQuestion}</p>
              </blockquote>
              <p>{cfrIntro.plain}</p>
              <p className="research-disclaimer" role="note">
                {cfrIntro.clarify}
              </p>
            </div>
            <aside className="cfr-hero__aside" aria-label="Exhibit map">
              <p className="mono research-banner__label">On this page</p>
              <ol className="cfr-toc">
                {cfrNav.map((item) => (
                  <li key={item.href}>
                    <a href={item.href}>{item.label}</a>
                  </li>
                ))}
              </ol>
            </aside>
          </div>
        </section>

        <section className="section" id="regret-matching" aria-labelledby="regret-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Interactive</p>
              <h2 id="regret-title">{regretMatchingCopy.title}</h2>
              <p>{regretMatchingCopy.summary}</p>
            </header>
            <RegretMatchingDemo />
          </div>
        </section>

        <section className="section" id="kuhn" aria-labelledby="kuhn-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Game</p>
              <h2 id="kuhn-title">{kuhnCopy.title}</h2>
              <p>{kuhnCopy.summary}</p>
            </header>
            <KuhnDiagram />
          </div>
        </section>

        <section className="section" id="training" aria-labelledby="training-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Measured</p>
              <h2 id="training-title">{trainingCopy.title}</h2>
              <p>{trainingCopy.summary}</p>
            </header>
            <TrainingTimeline />
          </div>
        </section>

        <section className="section" id="average" aria-labelledby="average-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Concepts</p>
              <h2 id="average-title">{averageCopy.title}</h2>
              <p>{averageCopy.summary}</p>
            </header>
            <ul className="diff-grid">
              {averageCopy.distinctions.map((item) => (
                <li key={item.title} className="diff-card">
                  <h3>{item.title}</h3>
                  <p>{item.detail}</p>
                </li>
              ))}
            </ul>
            <details className="math-details">
              <summary>{averageCopy.mathTitle}</summary>
              <ul className="math-list mono">
                {averageCopy.math.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </details>
          </div>
        </section>

        <section className="section" id="verification" aria-labelledby="verification-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Evidence</p>
              <h2 id="verification-title">{verificationCopy.title}</h2>
              <p>{verificationCopy.summary}</p>
            </header>
            <ul className="tech-grid">
              {verificationCopy.items.map((item) => (
                <li key={item.title} className="tech-card">
                  <h3>{item.title}</h3>
                  <p>{item.detail}</p>
                </li>
              ))}
            </ul>
            <div className="hero__actions">
              {verificationCopy.links.map((item) => (
                <Button key={item.href} href={item.href} variant="secondary" external>
                  {item.label}
                </Button>
              ))}
            </div>
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
              <Button href={continueCopy.secondary.href} variant="secondary" external>
                {continueCopy.secondary.label}
              </Button>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
