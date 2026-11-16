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
  /** How the action read at the table (set by the server). */
  label: ActionLabel;
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
  source: StrategySource;
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

export type StrategySource = "baseline_cfr" | "rnr" | "cfr_fallback";

/** A strategy version as served by `/strategy` and `/metrics` (certified metadata only). */
export interface StrategyVersion {
  version: number;
  source: StrategySource;
  rnr_p: number | null;
  certified_exploitability: number;
  exploitability_limit: number;
  digest: string;
  based_on_hands: number;
  installed_after_hand: number;
  /** Unix seconds. */
  installed_at: number;
}

/** Model-based quantities behind the active strategy (probabilistic, not verified). */
export interface ModelEstimates {
  prior: [string, number];
  hands_used: number;
  candidates: number;
  rule: string;
  score: number | null;
  posterior_gain_mean?: number;
  posterior_gain_sd?: number;
}

export interface TendencyAction {
  posterior_mean: number;
  interval_90: [number, number];
  observed: number;
  /** null until the player has been seen in this situation. */
  observed_frequency: number | null;
  population_prior_mean: number;
}

export interface Tendency {
  situation: string;
  description: string;
  observations: number;
  /** Weight of the population prior, in pseudo-observations. */
  prior_strength: number;
  /** Keyed by table label (check/bet or fold/call/raise), in the server's order. */
  actions: Record<string, TendencyAction>;
}

export interface UpdateStatus {
  running: boolean;
  running_job: { job_id: number; reason: string; based_on_hands: number; elapsed_seconds: number } | null;
  pending: boolean;
  recompute_every_hands: number;
  next_scheduled_after_hands: number;
  timeout_seconds: number;
  last_success: {
    job_id: number;
    version: number;
    reason: string;
    based_on_hands: number;
    seconds: number;
    finished_at: number;
  } | null;
  last_failure: { job_id: number; kind: string; message: string; based_on_hands: number; at: number } | null;
  counters: Record<string, number>;
}

/** `GET /sessions/{id}/metrics`. */
export interface SessionMetrics {
  session_id: string;
  hands_played: number;
  verified: {
    active_strategy: StrategyVersion & { certified_by: string };
    exploitability_limit: number;
    baseline_cfr_exploitability: number;
    budget_delta: number;
    observation_counts: {
      hands_observed: number;
      hands_in_model: number;
      human_decisions_in_model: number;
      showdowns_in_model: number;
      change_resets_after_hands: number[];
    };
  };
  estimates: {
    disclaimer: string;
    opponent_tendencies: Tendency[];
    active_strategy_model_estimates: ModelEstimates | null;
    change_detector: { statistic: number; alarm_threshold: number };
  };
  strategy_updates: UpdateStatus;
}

/** `GET /sessions/{id}/strategy`. */
export interface StrategyStatus {
  session_id: string;
  active: StrategyVersion & { model_estimates: ModelEstimates | null };
  versions: StrategyVersion[];
  updates: UpdateStatus;
}

export interface ServerConfig {
  game: string;
  research_version: string;
  research_digest: string;
  baseline_cfr_exploitability: number;
  budget_delta: number;
  exploitability_limit: number;
}
