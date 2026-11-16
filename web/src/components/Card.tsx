/** A Leduc card from the server's name, e.g. "K1": rank J/Q/K and suit 0 or 1. */

const SUITS = ["♠", "♥"] as const;
const RANK_NAMES: Record<string, string> = { J: "Jack", Q: "Queen", K: "King" };

export function parseCard(name: string): { rank: string; suit: string; red: boolean } {
  const rank = name.slice(0, 1);
  const suitIndex = Number(name.slice(1)) === 1 ? 1 : 0;
  return { rank, suit: SUITS[suitIndex], red: suitIndex === 1 };
}

export function cardLabel(name: string): string {
  const { rank, suit } = parseCard(name);
  return `${RANK_NAMES[rank] ?? rank} of ${suit === "♠" ? "spades" : "hearts"}`;
}

interface CardProps {
  /** Card name from the server; omit for a face-down card. */
  card?: string | null;
  size?: "md" | "lg";
  /** Accessible description of a face-down card. */
  hiddenLabel?: string;
}

export function Card({ card, size = "md", hiddenLabel = "Face-down card" }: CardProps) {
  if (!card) {
    return (
      <div className={`card card-${size} card-back`} role="img" aria-label={hiddenLabel}>
        <span className="card-back-mark" aria-hidden="true">
          PM
        </span>
      </div>
    );
  }
  const { rank, suit, red } = parseCard(card);
  return (
    <div
      className={`card card-${size} card-face${red ? " card-red" : ""}`}
      role="img"
      aria-label={cardLabel(card)}
      data-card={card}
    >
      <span className="card-corner" aria-hidden="true">
        {rank}
        <br />
        {suit}
      </span>
      <span className="card-center" aria-hidden="true">
        {rank}
      </span>
      <span className="card-corner card-corner-bottom" aria-hidden="true">
        {rank}
        <br />
        {suit}
      </span>
    </div>
  );
}

/** Empty slot where the public card will be dealt. */
export function CardSlot({ size = "md", label }: { size?: "md" | "lg"; label: string }) {
  return (
    <div className={`card card-${size} card-slot`} role="img" aria-label={label}>
      <span aria-hidden="true">?</span>
    </div>
  );
}
