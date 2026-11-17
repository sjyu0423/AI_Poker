import { HIDDEN_CARD_EXAMPLE } from "@/content/m5OpponentModeling";
import { hiddenCopy } from "@/content/opponentModelingPage";

function PolicyBars({
  title,
  detail,
  rates,
}: {
  title: string;
  detail: string;
  rates: { label: string; value: number }[];
}) {
  return (
    <div className="hidden-card">
      <h3>{title}</h3>
      <p>{detail}</p>
      <ul>
        {rates.map((row) => (
          <li key={row.label}>
            <div className="strategy-card__row">
              <span>{row.label}</span>
              <span className="mono">{(row.value * 100).toFixed(0)}%</span>
            </div>
            <div className="prob-bar">
              <i style={{ width: `${row.value * 100}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function HiddenCardDiagram() {
  const example = HIDDEN_CARD_EXAMPLE;
  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Illustrative · not a live simulation</p>
      </div>
      <p>
        Public betting frequency ≈{" "}
        <strong className="mono">{(example.publicBetRate * 100).toFixed(0)}%</strong> for both
        styles below—yet the card-conditional policies differ sharply.
      </p>
      <div className="hidden-grid">
        <PolicyBars
          title={example.alwaysMix.name}
          detail={example.alwaysMix.detail}
          rates={[
            { label: "Bet with J", value: example.alwaysMix.jBet },
            { label: "Bet with Q", value: example.alwaysMix.qBet },
            { label: "Bet with K", value: example.alwaysMix.kBet },
          ]}
        />
        <PolicyBars
          title={example.polarized.name}
          detail={example.polarized.detail}
          rates={[
            { label: "Bet with J", value: example.polarized.jBet },
            { label: "Bet with Q", value: example.polarized.qBet },
            { label: "Bet with K", value: example.polarized.kBet },
          ]}
        />
      </div>
      <p className="research-disclaimer" role="note">
        {example.note}
      </p>
      <p className="muted">{hiddenCopy.summary}</p>
    </div>
  );
}
