import { links, siteMeta, limitations } from "@/content/site";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="shell site-footer__grid">
        <div>
          <p className="site-footer__brand">{siteMeta.name}</p>
          <p className="muted">
            Research showcase for v{siteMeta.version}. Local-first Leduc Hold&apos;em AI
            with exact exploitability budgets.
          </p>
        </div>
        <div>
          <p className="site-footer__heading">Links</p>
          <ul className="site-footer__links">
            <li>
              <a href={links.github} target="_blank" rel="noopener noreferrer">
                Repository
              </a>
            </li>
            <li>
              <a href={links.release} target="_blank" rel="noopener noreferrer">
                Release v1.0.0
              </a>
            </li>
            <li>
              <a href={links.overview} target="_blank" rel="noopener noreferrer">
                Overview
              </a>
            </li>
            <li>
              <a href={links.researchFreeze} target="_blank" rel="noopener noreferrer">
                Research freeze
              </a>
            </li>
          </ul>
        </div>
        <div>
          <p className="site-footer__heading">Known limits</p>
          <ul className="site-footer__limits">
            {limitations.slice(0, 3).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
      <div className="shell site-footer__legal muted">
        <p>
          Not an online casino. No real-money play. Experiments used simulated Leduc
          opponents unless marked as exact guarantees.
        </p>
      </div>
    </footer>
  );
}
