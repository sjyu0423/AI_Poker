import type { HandView } from "@/lib/types";

import { Card, CardSlot } from "./Card";
import { chips, describeEvent, resultHeadline } from "./format";

/** The poker table for one hand. Everything shown comes from the server's hand view. */
export function Table({ hand }: { hand: HandView }) {
  const complete = hand.status === "complete";
  const result = hand.result;
  return (
    <section className="table" aria-label={`Hand ${hand.hand_number}`}>
      <div className="seat seat-ai">
        <div className="seat-label">
          <span className="seat-name">PokerMind</span>
          <span className="seat-chips">in pot: {hand.contributions.ai}</span>
        </div>
        <div className="seat-cards">
          {result?.ai_card ? (
            <Card card={result.ai_card} />
          ) : (
            <Card hiddenLabel="PokerMind's private card (face down)" />
          )}
        </div>
        {hand.to_act === "ai" && <span className="turn-chip">thinking…</span>}
      </div>

      <div className="board">
        <div className="round-indicator" aria-live="polite">
          <span className={hand.round === 1 ? "round-dot active" : "round-dot done"}>Round 1</span>
          <span className={hand.round === 2 ? "round-dot active" : "round-dot"}>Round 2</span>
        </div>
        <div className="public-card">
          <span className="zone-label">Public card</span>
          {hand.public_card ? (
            <Card card={hand.public_card} />
          ) : (
            <CardSlot label="Public card: dealt after round 1" />
          )}
          {!hand.public_card && <span className="zone-hint">dealt after round 1</span>}
        </div>
        <div className="pot" aria-live="polite">
          <span className="pot-label">Pot</span>
          <span className="pot-value" data-testid="pot">
            {hand.pot}
          </span>
        </div>
      </div>

      <div className="seat seat-human">
        <div className="seat-cards">
          <Card card={hand.human_card} size="lg" />
        </div>
        <div className="seat-label">
          <span className="seat-name">You</span>
          <span className="seat-chips">in pot: {hand.contributions.human}</span>
          <span className="zone-hint">your private card</span>
        </div>
      </div>

      <ol className="action-log" aria-label="Actions this hand">
        <li className="log-meta">
          Hand {hand.hand_number} · {hand.first_to_act === "human" ? "you act" : "PokerMind acts"}{" "}
          first
        </li>
        {hand.actions.map((e, i) => (
          <li key={i} className={`log-${e.actor}`}>
            <span className="log-round">R{e.round}</span> {describeEvent(e)}
          </li>
        ))}
      </ol>

      {complete && result && (
        <div
          className={`result result-${result.winner ?? "split"}`}
          role="status"
          data-testid="result"
        >
          <strong>{resultHeadline(result)}</strong>
          <span>
            {result.human_net === 0
              ? "No chips change hands."
              : `You ${result.human_net > 0 ? "won" : "lost"} ${Math.abs(result.human_net)} chip${Math.abs(result.human_net) === 1 ? "" : "s"} (${chips(result.human_net)}).`}
          </span>
          {result.ended_by === "fold" && (
            <span className="muted">PokerMind&apos;s card stays hidden after a fold.</span>
          )}
        </div>
      )}
    </section>
  );
}
