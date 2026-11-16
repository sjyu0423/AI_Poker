import type { ActionEvent, HandResult } from "@/lib/types";

export function chips(n: number): string {
  const value = Math.abs(n) < 1e-9 ? 0 : n;
  return `${value > 0 ? "+" : ""}${value}`;
}

/** Past actions as the server reports them. The server's history does not distinguish a
 * check from a call, or a bet from a raise, so neither does this text. */
const WORDS: Record<ActionEvent["action"], [string, string]> = {
  fold: ["fold", "folds"],
  call: ["check/call", "checks/calls"],
  raise: ["bet/raise", "bets/raises"],
};

export function describeEvent(e: ActionEvent): string {
  const [you, they] = WORDS[e.action];
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
