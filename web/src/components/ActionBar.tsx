"use client";

import { useEffect, useMemo } from "react";

import type { ActionLabel, ActionName, HandView } from "@/lib/types";

/** Keyboard shortcut for each label the server can send. */
export const SHORTCUTS: Record<ActionLabel, string> = {
  fold: "f",
  check: "k",
  call: "c",
  bet: "b",
  raise: "r",
};

const TITLES: Record<ActionLabel, string> = {
  fold: "Fold",
  check: "Check",
  call: "Call",
  bet: "Bet",
  raise: "Raise",
};

interface ActionBarProps {
  hand: HandView | null;
  /** A request is in flight. */
  pending: boolean;
  /** Controls unusable for another reason (e.g. the server is unreachable). */
  disabled?: boolean;
  onAct: (action: ActionName) => void;
  onDeal: () => void;
}

function typingTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
  );
}

/** Buttons for exactly the legal actions the server listed, or the next-hand button. */
export function ActionBar({ hand, pending, disabled = false, onAct, onDeal }: ActionBarProps) {
  const locked = pending || disabled;
  const yourTurn = hand?.status === "awaiting_human" && hand.to_act === "human";
  const canDeal = hand === null || hand.status === "complete";
  const legal = useMemo(() => (yourTurn && hand ? hand.legal_actions : []), [yourTurn, hand]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      if (typingTarget(event.target) || locked) return;
      const key = event.key.toLowerCase();
      // Enter on a focused button already activates it; N works from anywhere.
      const enterElsewhere = key === "enter" && !(event.target instanceof HTMLButtonElement);
      if (canDeal && (key === "n" || enterElsewhere)) {
        event.preventDefault();
        onDeal();
        return;
      }
      const match = legal.find((a) => SHORTCUTS[a.label] === key);
      if (match) {
        event.preventDefault();
        onAct(match.action);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [legal, canDeal, locked, onAct, onDeal]);

  if (canDeal) {
    return (
      <div className="action-bar">
        <button
          type="button"
          className="btn btn-primary btn-wide"
          onClick={onDeal}
          disabled={locked}
          aria-keyshortcuts="N"
        >
          {pending ? "Dealing…" : hand === null ? "Deal first hand" : "Next hand"}
          <kbd aria-hidden="true">N</kbd>
        </button>
      </div>
    );
  }

  return (
    <div className="action-bar" role="group" aria-label="Your actions">
      {legal.map((a) => (
        <button
          key={a.action}
          type="button"
          className={`btn btn-action btn-${a.label}`}
          onClick={() => onAct(a.action)}
          disabled={locked}
          aria-keyshortcuts={SHORTCUTS[a.label].toUpperCase()}
        >
          {TITLES[a.label]}
          <kbd aria-hidden="true">{SHORTCUTS[a.label].toUpperCase()}</kbd>
        </button>
      ))}
      {!yourTurn && <p className="muted">Waiting for PokerMind…</p>}
      {pending && (
        <span className="sr-only" role="status">
          Sending your action…
        </span>
      )}
    </div>
  );
}
