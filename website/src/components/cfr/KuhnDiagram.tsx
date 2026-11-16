import { kuhnCopy } from "@/content/cfrPage";

export function KuhnDiagram() {
  return (
    <div className="kuhn-block">
      <div className="kuhn-cards" aria-label="Kuhn Poker deck">
        {kuhnCopy.cards.map((card) => (
          <div key={card.rank} className="kuhn-card">
            <span className="kuhn-card__rank">{card.rank}</span>
            <span className="kuhn-card__name">{card.name}</span>
            <span className="muted">{card.note}</span>
          </div>
        ))}
      </div>

      <ul className="kuhn-points">
        {kuhnCopy.points.map((point) => (
          <li key={point.title}>
            <h3>{point.title}</h3>
            <p>{point.detail}</p>
          </li>
        ))}
      </ul>

      <figure className="kuhn-tree" aria-labelledby="kuhn-tree-caption">
        <svg viewBox="0 0 640 280" role="img" aria-hidden="true" className="kuhn-tree__svg">
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" />
            </marker>
          </defs>
          <text x="320" y="28" textAnchor="middle" className="kuhn-tree__label">
            Info set Q|xb — “I hold Queen; check, then bet”
          </text>
          <rect x="250" y="48" width="140" height="52" rx="10" className="kuhn-tree__node" />
          <text x="320" y="72" textAnchor="middle" className="kuhn-tree__node-text">
            Acting player
          </text>
          <text x="320" y="90" textAnchor="middle" className="kuhn-tree__node-sub">
            sees Q only
          </text>

          <line x1="290" y1="100" x2="160" y2="150" className="kuhn-tree__edge" markerEnd="url(#arrow)" />
          <line x1="350" y1="100" x2="480" y2="150" className="kuhn-tree__edge" markerEnd="url(#arrow)" />

          <rect x="70" y="158" width="180" height="70" rx="10" className="kuhn-tree__leaf" />
          <text x="160" y="188" textAnchor="middle" className="kuhn-tree__node-text">
            Hidden history
          </text>
          <text x="160" y="208" textAnchor="middle" className="kuhn-tree__node-sub">
            Opponent holds J
          </text>

          <rect x="390" y="158" width="180" height="70" rx="10" className="kuhn-tree__leaf" />
          <text x="480" y="188" textAnchor="middle" className="kuhn-tree__node-text">
            Hidden history
          </text>
          <text x="480" y="208" textAnchor="middle" className="kuhn-tree__node-sub">
            Opponent holds K
          </text>

          <text x="320" y="258" textAnchor="middle" className="kuhn-tree__note">
            Same information set · two indistinguishable worlds
          </text>
        </svg>
        <figcaption id="kuhn-tree-caption">{kuhnCopy.treeCaption}</figcaption>
      </figure>
    </div>
  );
}
