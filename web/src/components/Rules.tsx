/** Short Leduc Hold'em rules (as implemented by the server, docs/games/leduc.md). */
export function Rules() {
  return (
    <details className="panel rules">
      <summary className="panel-title">How Leduc Hold&apos;em works</summary>
      <ul>
        <li>
          The deck has six cards: two Jacks, two Queens and two Kings. Suits never matter.
        </li>
        <li>
          Each player antes 1 chip and gets one <strong>private</strong> card that only they see.
        </li>
        <li>
          <strong>Round 1:</strong> bets and raises are 2 chips. Then one{" "}
          <strong>public</strong> card is dealt face up for both players.
        </li>
        <li>
          <strong>Round 2:</strong> bets and raises are 4 chips. At most two bets or raises per
          round.
        </li>
        <li>
          At showdown, a private card that <strong>pairs</strong> the public card wins.
          Otherwise the higher card wins (K &gt; Q &gt; J). Equal ranks split the pot.
        </li>
        <li>You and PokerMind alternate who acts first each hand.</li>
      </ul>
    </details>
  );
}
