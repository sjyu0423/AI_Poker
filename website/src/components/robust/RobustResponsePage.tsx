import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Button } from "@/components/Button";
import { DangerPanel } from "@/components/robust/DangerPanel";
import { RnrConstructionDial } from "@/components/robust/RnrConstructionDial";
import { ExploitabilityFrontier } from "@/components/robust/ExploitabilityFrontier";
import { PersonalizationPanel } from "@/components/robust/PersonalizationPanel";
import { ReliabilityPanel } from "@/components/robust/ReliabilityPanel";
import { TechnicalDeepDive } from "@/components/robust/TechnicalDeepDive";
import {
  closingCopy,
  dangerCopy,
  frontierCopy,
  personalCopy,
  reliabilityCopy,
  rnrCopy,
  robustIntro,
  robustNav,
  whyCopy,
  deepDiveCopy,
} from "@/content/robustResponsePage";
import { withBasePath } from "@/lib/paths";

export function RobustResponsePage() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <SiteHeader
        brandHref={withBasePath("/")}
        navItems={[...robustNav]}
        trailingHref={withBasePath("/")}
        trailingLabel="Home"
      />
      <main id="main" className="robust-page">
        <section className="section cfr-hero" id="intro" aria-labelledby="robust-title">
          <div className="shell cfr-hero__grid">
            <div>
              <p className="section-kicker">{robustIntro.kicker}</p>
              <h1 id="robust-title">{robustIntro.headline}</h1>
              <p className="cfr-hero__lede">{robustIntro.lede}</p>
              <p>{robustIntro.plain}</p>
            </div>
            <aside className="cfr-hero__aside" aria-label="Exhibit map">
              <p className="mono research-banner__label">On this page</p>
              <ol className="cfr-toc">
                {robustNav.map((item) => (
                  <li key={item.href}>
                    <a href={item.href}>{item.label}</a>
                  </li>
                ))}
              </ol>
            </aside>
          </div>
        </section>

        <section className="section" id="danger" aria-labelledby="danger-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Motivation</p>
              <h2 id="danger-title">{dangerCopy.title}</h2>
            </header>
            <DangerPanel />
          </div>
        </section>

        <section className="section" id="rnr" aria-labelledby="rnr-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Construction</p>
              <h2 id="rnr-title">{rnrCopy.title}</h2>
            </header>
            <RnrConstructionDial />
          </div>
        </section>

        <section className="section" id="frontier" aria-labelledby="frontier-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Interactive</p>
              <h2 id="frontier-title">{frontierCopy.title}</h2>
            </header>
            <ExploitabilityFrontier />
          </div>
        </section>

        <section className="section" id="why" aria-labelledby="why-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Decision</p>
              <h2 id="why-title">{whyCopy.title}</h2>
            </header>
            <div className="why-grid">
              {whyCopy.bullets.map((item) => (
                <article key={item.title} className="why-card">
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="section" id="personal" aria-labelledby="personal-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">Adaptation</p>
              <h2 id="personal-title">{personalCopy.title}</h2>
            </header>
            <PersonalizationPanel />
          </div>
        </section>

        <section className="section" id="reliability" aria-labelledby="reliability-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">M7B</p>
              <h2 id="reliability-title">{reliabilityCopy.title}</h2>
            </header>
            <ReliabilityPanel />
          </div>
        </section>

        <section className="section" id="deep" aria-labelledby="deep-title">
          <div className="shell">
            <header className="section-header">
              <p className="section-kicker">For reviewers</p>
              <h2 id="deep-title">{deepDiveCopy.title}</h2>
            </header>
            <TechnicalDeepDive />
          </div>
        </section>

        <section className="section final-cta" aria-labelledby="closing-title">
          <div className="shell final-cta__panel">
            <h2 id="closing-title">{closingCopy.headline}</h2>
            <p>{closingCopy.body}</p>
            <ol className="closing-stack">
              {closingCopy.steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
            <div className="hero__actions">
              <Button href={withBasePath(closingCopy.primary.href)}>
                {closingCopy.primary.label}
              </Button>
              <Button href={withBasePath(closingCopy.secondary.href)} variant="secondary">
                {closingCopy.secondary.label}
              </Button>
              <Button href={withBasePath(closingCopy.tertiary.href)} variant="secondary">
                {closingCopy.tertiary.label}
              </Button>
              <Button href={closingCopy.docs.href} variant="ghost" external>
                {closingCopy.docs.label}
              </Button>
              <Button href={closingCopy.app.href} variant="ghost" external>
                {closingCopy.app.label}
              </Button>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
