"use client";

import { useSyncExternalStore } from "react";

const KEY = "pokermind.guide-dismissed";

function subscribe(onChange: () => void) {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

/** First-time guidance, dismissible (remembered in this browser). */
export function Guide() {
  const dismissed = useSyncExternalStore(
    subscribe,
    () => window.localStorage.getItem(KEY) === "1",
    () => true, // server render: hidden
  );
  if (dismissed) return null;
  return (
    <aside className="guide" aria-labelledby="guide-title">
      <h2 id="guide-title">New here?</h2>
      <ol>
        <li>Deal a hand. Your private card is at the bottom; PokerMind&apos;s is face down.</li>
        <li>Choose one of the buttons. Only the moves allowed right now are shown.</li>
        <li>After round 1, the public card appears in the middle.</li>
        <li>Keyboard: K check · B bet · C call · R raise · F fold · N next hand.</li>
      </ol>
      <button
        type="button"
        className="btn btn-ghost"
        onClick={() => {
          window.localStorage.setItem(KEY, "1");
          window.dispatchEvent(new StorageEvent("storage", { key: KEY }));
        }}
      >
        Got it
      </button>
    </aside>
  );
}
