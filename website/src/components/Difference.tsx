import { differentiators } from "@/content/site";

export function Difference() {
  return (
    <section className="section" id="difference" aria-labelledby="difference-title">
      <div className="shell">
        <header className="section-header">
          <p className="section-kicker">Capabilities</p>
          <h2 id="difference-title">What makes PokerMind different?</h2>
          <p>
            Three ideas work together: learn what an opponent tends to do, adapt
            carefully, and prove the new strategy still sits inside a hard safety
            limit.
          </p>
        </header>
        <ol className="diff-grid">
          {differentiators.map((item, index) => (
            <li key={item.id} className="diff-card">
              <span className="diff-card__index mono" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </span>
              <h3>{item.title}</h3>
              <p>{item.plain}</p>
              <p className="diff-card__tech muted">{item.technical}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
