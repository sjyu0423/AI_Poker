import { technology, limitations, links } from "@/content/site";

export function Technology() {
  return (
    <section className="section tech" id="technology" aria-labelledby="tech-title">
      <div className="shell">
        <header className="section-header">
          <p className="section-kicker">Stack</p>
          <h2 id="tech-title">Technology and architecture</h2>
          <p>
            Production play uses tabular CFR/RNR strategies with exact certification.
            Neural networks appear only in research experiments that measured—and
            rejected—plain distillation for Leduc.
          </p>
        </header>

        <ul className="tech-grid">
          {technology.map((item) => (
            <li key={item.name} className="tech-card">
              <h3>{item.name}</h3>
              <p>{item.role}</p>
            </li>
          ))}
        </ul>

        <div className="tech-notes">
          <div>
            <h3>Architecture in brief</h3>
            <p>
              A FastAPI backend owns the deck, legality, settlement, and observations.
              The Next.js UI renders server state only. Background RNR re-solves install
              new strategies only after exact re-certification. Settings come from one
              digest-checked research configuration.
            </p>
            <p>
              <a href={links.appBackend} target="_blank" rel="noopener noreferrer">
                API and backend design →
              </a>
            </p>
          </div>
          <div>
            <h3>Known limitations</h3>
            <ul className="limit-list">
              {limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
