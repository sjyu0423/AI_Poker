/**
 * Response types of the PokerMind API (version 1).
 *
 * They mirror the views built in `src/pokermind/app/session.py` and documented in
 * `docs/design/app_backend.md`. Only fields the backend actually returns are declared.
 * The frontend never derives game facts (legal actions, hidden cards, payouts) itself.
 */

export type ActionName = "fold" | "call" | "raise";
export type ActionLabel = "fold" | "check" | "call" | "bet" | "raise";
export type Actor = "human" | "ai";

export interface LegalAction {
  action: ActionName;
  label: ActionLabel;
}

export interface ActionEvent {
  actor: Actor;
  round: 1 | 2;
  action: ActionName;
}

export interface HandResult {
  ended_by: "fold" | "showdown";
  /** null is a split pot. */
  winner: Actor | null;
  human_net: number;
  ai_net: number;
  pot: number;
  /** The AI's card. Present only after a showdown, otherwise null. */
  ai_card: string | null;
}

export interface HandView {
  hand_number: number;
  status: "awaiting_human" | "complete";
  human_seat: 0 | 1;
  first_to_act: Actor;
  human_card: string;
  public_card: string | null;
  round: 1 | 2;
  actions: ActionEvent[];
  contributions: { human: number; ai: number };
  pot: number;
  to_act: Actor | null;
  legal_actions: LegalAction[];
  strategy_version: number;
  result: HandResult | null;
}

export interface SessionStats {
  hands_played: number;
  human_net_chips: number;
  ai_net_chips: number;
  human_mean_per_hand: number;
  human_mean_standard_error: number | null;
  human_wins: number;
  ai_wins: number;
  splits: number;
  showdowns: number;
  human_folds: number;
  ai_folds: number;
}

export interface StrategySummary {
  active_version: number;
  source: "baseline_cfr" | "rnr" | "cfr_fallback";
  certified_exploitability: number;
  exploitability_limit: number;
  recomputation_running: boolean;
}

export interface SessionState {
  session_id: string;
  state_version: number;
  hands_played: number;
  hand: HandView | null;
  stats: SessionStats;
  strategy: StrategySummary;
}

export interface HistoryPage {
  session_id: string;
  total: number;
  offset: number;
  hands: HandView[];
}

/** The parts of `GET /sessions/{id}/metrics` this UI reads (M8C will use more). */
export interface SessionMetrics {
  session_id: string;
  hands_played: number;
  verified: {
    active_strategy: {
      version: number;
      source: StrategySummary["source"];
      certified_exploitability: number;
      exploitability_limit: number;
      based_on_hands: number;
      installed_after_hand: number;
    };
    exploitability_limit: number;
    observation_counts: {
      hands_observed: number;
      hands_in_model: number;
      human_decisions_in_model: number;
      showdowns_in_model: number;
      change_resets_after_hands: number[];
    };
  };
  strategy_updates: {
    running: boolean;
    pending: boolean;
    recompute_every_hands: number;
    next_scheduled_after_hands: number;
    last_success: { version: number; based_on_hands: number; finished_at: number } | null;
    last_failure: { kind: string; message: string } | null;
  };
}

export interface ServerConfig {
  game: string;
  research_version: string;
  research_digest: string;
  baseline_cfr_exploitability: number;
  budget_delta: number;
  exploitability_limit: number;
}
