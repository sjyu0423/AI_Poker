import { deepDiveCopy } from "@/content/robustResponsePage";

export function TechnicalDeepDive() {
  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Optional · technical</p>
      </div>
      <p className="muted">{deepDiveCopy.summary}</p>

      <div className="deep-list">
        {deepDiveCopy.items.map((item) => (
          <details key={item.title} className="deep-item">
            <summary>{item.title}</summary>
            <p>{item.body}</p>
          </details>
        ))}
      </div>

      <div className="hero__actions" style={{ marginTop: "1.25rem" }}>
        <a className="btn btn--secondary" href={deepDiveCopy.implLink.href} target="_blank" rel="noopener noreferrer">
          {deepDiveCopy.implLink.label}
        </a>
        <a className="btn btn--ghost" href={deepDiveCopy.designLink.href} target="_blank" rel="noopener noreferrer">
          {deepDiveCopy.designLink.label}
        </a>
        <a className="btn btn--ghost" href={deepDiveCopy.freezeLink.href} target="_blank" rel="noopener noreferrer">
          {deepDiveCopy.freezeLink.label}
        </a>
      </div>
    </div>
  );
}
