export function HeroVisual() {
  return (
    <div className="hero-visual" aria-hidden="true">
      <div className="hero-visual__stage">
        <div className="hero-card hero-card--back hero-card--left">
          <span className="hero-card__pip">♠</span>
        </div>
        <div className="hero-card hero-card--face hero-card--center">
          <span className="hero-card__rank">K</span>
          <span className="hero-card__suit">♦</span>
          <div className="hero-card__bars">
            <div className="hero-bar">
              <span>Check</span>
              <i style={{ width: "28%" }} />
            </div>
            <div className="hero-bar">
              <span>Bet</span>
              <i style={{ width: "72%" }} />
            </div>
          </div>
        </div>
        <div className="hero-card hero-card--face hero-card--right">
          <span className="hero-card__rank">Q</span>
          <span className="hero-card__suit">♣</span>
          <div className="hero-card__bars">
            <div className="hero-bar">
              <span>Fold</span>
              <i style={{ width: "18%" }} />
            </div>
            <div className="hero-bar">
              <span>Call</span>
              <i style={{ width: "41%" }} />
            </div>
            <div className="hero-bar">
              <span>Raise</span>
              <i style={{ width: "41%" }} />
            </div>
          </div>
        </div>
        <div className="hero-badge">
          <span className="mono">expl ≤ 0.1030</span>
          <strong>Certified budget</strong>
        </div>
      </div>
      <p className="hero-visual__caption muted">
        Illustrative strategy probabilities—not live play on this page.
      </p>
    </div>
  );
}
