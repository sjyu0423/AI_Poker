import type { ActionEvent, HandResult } from "@/lib/types";

export function chips(n: number): string {
  const value = Math.abs(n) < 1e-9 ? 0 : n;
  return `${value > 0 ? "+" : ""}${value}`;
}

/** A past action as it read at the table (the server's label). */
const WORDS: Record<ActionEvent["label"], [string, string]> = {
  fold: ["fold", "folds"],
  check: ["check", "checks"],
  call: ["call", "calls"],
  bet: ["bet", "bets"],
  raise: ["raise", "raises"],
};

export function describeEvent(e: ActionEvent): string {
  const [you, they] = WORDS[e.label];
  return e.actor === "human" ? `You ${you}` : `PokerMind ${they}`;
}

export function resultHeadline(result: HandResult): string {
  if (result.winner === null) return "Split pot";
  const how = result.ended_by === "fold" ? "by fold" : "at showdown";
  return result.winner === "human" ? `You win ${how}` : `PokerMind wins ${how}`;
}

export function exploitability(x: number): string {
  return `${x.toFixed(4)} chips/hand`;
}

export function percent(x: number): string {
  return `${Math.round(x * 100)}%`;
}
