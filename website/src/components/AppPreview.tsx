import { screenshots, links } from "@/content/site";
import { Button } from "@/components/Button";
import { withBasePath } from "@/lib/paths";

export function AppPreview() {
  const featured = screenshots.filter((shot) =>
    [
      "screenshots/02-first-hand.png",
      "screenshots/06-dashboard-tendencies.png",
      "screenshots/07-dashboard-strategy.png",
      "screenshots/04-learning.png",
    ].includes(shot.src),
  );

  return (
    <section className="section preview" id="preview" aria-labelledby="preview-title">
      <div className="shell">
        <header className="section-header">
          <p className="section-kicker">Application</p>
          <h2 id="preview-title">Application preview</h2>
          <p>
            Real screenshots from a local demo session. The playable interface,
            learning dashboard, strategy history, and certified exploitability
            display all run on your machine—not on this static site.
          </p>
        </header>

        <div className="preview-feature">
          {featured.map((shot) => (
            <figure key={shot.src} className="preview-shot">
              {/* Plain img: next/image omitted basePath under static export. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={withBasePath(shot.src)} alt={shot.alt} width={1440} height={1000} />
              <figcaption>{shot.caption}</figcaption>
            </figure>
          ))}
        </div>

        <details className="preview-more">
          <summary>More session screenshots</summary>
          <div className="preview-feature preview-feature--compact">
            {screenshots
              .filter((shot) => !featured.includes(shot))
              .map((shot) => (
                <figure key={shot.src} className="preview-shot">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={withBasePath(shot.src)} alt={shot.alt} width={1440} height={1000} />
                  <figcaption>{shot.caption}</figcaption>
                </figure>
              ))}
          </div>
        </details>

        <div className="preview-cta">
          <p>
            Install from the published release, train the frozen CFR hero once, then
            launch locally with <code className="mono">pokermind launch</code>.
          </p>
          <div className="hero__actions">
            <Button href={links.release} external>
              GitHub release
            </Button>
            <Button href={links.quickStart} variant="secondary" external>
              Quick start
            </Button>
            <Button href={links.demo} variant="ghost" external>
              Demo guide
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
