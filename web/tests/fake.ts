/** A scripted stand-in for the FastAPI backend, injected through PokerMindApi's fetch. */

import { PokerMindApi } from "@/lib/api";
import type {
  HandView,
  SessionMetrics,
  SessionState,
  StrategyStatus,
  StrategyVersion,
  Tendency,
  TendencyAction,
  UpdateStatus,
} from "@/lib/types";

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

type Plain = { status: number; body?: unknown } | "network-error";
type Reply = Plain | Promise<Plain>;
type Handler = (call: Call) => Reply;

export function hand(overrides: Partial<HandView> = {}): HandView {
  return {
    hand_number: 1,
    status: "awaiting_human",
    human_seat: 0,
    first_to_act: "human",
    human_card: "K1",
    public_card: null,
    round: 1,
    actions: [],
    contributions: { human: 1, ai: 1 },
    pot: 2,
    to_act: "human",
    legal_actions: [
      { action: "call", label: "check" },
      { action: "raise", label: "bet" },
    ],
    strategy_version: 1,
    result: null,
    ...overrides,
  };
}

export function state(overrides: Partial<SessionState> = {}): SessionState {
  return {
    session_id: "s1",
    state_version: 0,
    hands_played: 0,
    hand: null,
    stats: {
      hands_played: 0,
      human_net_chips: 0,
      ai_net_chips: 0,
      human_mean_per_hand: 0,
      human_mean_standard_error: null,
      human_wins: 0,
      ai_wins: 0,
      splits: 0,
      showdowns: 0,
      human_folds: 0,
      ai_folds: 0,
    },
    strategy: {
      active_version: 1,
      source: "baseline_cfr",
      certified_exploitability: 0.0029968965501324385,
      exploitability_limit: 0.10299689655013244,
      recomputation_running: false,
    },
    ...overrides,
  };
}

const PRIOR: Record<string, Record<string, number>> = {
  "R1:cr": { check: 0.6, bet: 0.4 },
  "R1:fcr": { fold: 0.38, call: 0.44, raise: 0.18 },
  "R1:fc": { fold: 0.3, call: 0.7 },
  "R2:cr": { check: 0.55, bet: 0.45 },
  "R2:fcr": { fold: 0.35, call: 0.45, raise: 0.2 },
  "R2:fc": { fold: 0.25, call: 0.75 },
};

/** Tendencies as the backend computes them: Dirichlet(observed + 64 × prior), 90% intervals
 * approximated here (tests check display, not the Beta quantiles). */
export function tendencies(observed: Record<string, Record<string, number>> = {}): Tendency[] {
  return Object.entries(PRIOR).map(([situation, prior]) => {
    const seen = observed[situation] ?? {};
    const n = Object.values(seen).reduce((a, b) => a + b, 0);
    const actions: Record<string, TendencyAction> = {};
    for (const [label, p] of Object.entries(prior)) {
      const k = seen[label] ?? 0;
      const mean = (k + 64 * p) / (n + 64);
      actions[label] = {
        posterior_mean: mean,
        interval_90: [Math.max(0, mean - 0.09), Math.min(1, mean + 0.09)],
        observed: k,
        observed_frequency: n > 0 ? k / n : null,
        population_prior_mean: p,
      };
    }
    return { situation, description: `situation ${situation}`, observations: n, prior_strength: 64, actions };
  });
}

export function version(overrides: Partial<StrategyVersion> = {}): StrategyVersion {
  return {
    version: 1,
    source: "baseline_cfr",
    rnr_p: null,
    certified_exploitability: 0.0029968965501324385,
    exploitability_limit: 0.10299689655013244,
    digest: "3f1136d4",
    based_on_hands: 0,
    installed_after_hand: 0,
    installed_at: 1_794_800_000,
    ...overrides,
  };
}

export function updates(overrides: Partial<UpdateStatus> = {}): UpdateStatus {
  return {
    running: false,
    running_job: null,
    pending: false,
    recompute_every_hands: 50,
    next_scheduled_after_hands: 50,
    timeout_seconds: 600,
    last_success: null,
    last_failure: null,
    counters: { started: 0, installed: 0, failed: 0, timed_out: 0, cancelled: 0, discarded_stale: 0 },
    ...overrides,
  };
}

export function metrics(
  handsObserved = 0,
  running = false,
  extra: {
    observed?: Record<string, Record<string, number>>;
    active?: StrategyVersion;
    updates?: Partial<UpdateStatus>;
    resets?: number[];
  } = {},
): SessionMetrics {
  const active = extra.active ?? version();
  return {
    session_id: "s1",
    hands_played: handsObserved,
    verified: {
      active_strategy: { ...active, certified_by: "M3 recursive exact evaluator" },
      exploitability_limit: 0.10299689655013244,
      baseline_cfr_exploitability: 0.0029968965501324385,
      budget_delta: 0.1,
      observation_counts: {
        hands_observed: handsObserved,
        hands_in_model: handsObserved,
        human_decisions_in_model: 0,
        showdowns_in_model: 0,
        change_resets_after_hands: extra.resets ?? [],
      },
    },
    estimates: {
      disclaimer:
        "Probabilistic estimates from the AI's Bayesian opponent model. Their calibration was measured only against simulated development opponents (M7B); they are not calibrated for human players.",
      opponent_tendencies: tendencies(extra.observed),
      active_strategy_model_estimates: null,
      change_detector: { statistic: 0.1, alarm_threshold: 5 },
    },
    strategy_updates: updates({ running, ...extra.updates }),
  };
}

export class FakeServer {
  calls: Call[] = [];
  routes = new Map<string, Handler>();
  current: SessionState = state();
  metrics: SessionMetrics = metrics();
  historyHands: HandView[] = [];
  versions: StrategyVersion[] = [version()];

  constructor() {
    this.on("GET /health", () => ({ status: 200, body: { status: "ok", api_version: "1" } }));
    this.on("POST /sessions", () => ({ status: 201, body: this.current }));
    this.on("GET /sessions/s1", () => ({ status: 200, body: this.current }));
    this.on("DELETE /sessions/s1", () => ({ status: 204 }));
    this.on("GET /sessions/s1/metrics", () => ({ status: 200, body: this.metrics }));
    this.on("GET /sessions/s1/strategy", () => ({ status: 200, body: this.strategyStatus() }));
    this.on("GET /sessions/s1/history", () => ({
      status: 200,
      body: { session_id: "s1", total: this.historyHands.length, offset: 0, hands: this.historyHands },
    }));
  }

  strategyStatus(): StrategyStatus {
    const active = this.versions.at(-1) ?? version();
    return {
      session_id: "s1",
      active: { ...active, model_estimates: null },
      versions: this.versions,
      updates: this.metrics.strategy_updates,
    };
  }

  on(route: string, handler: Handler): void {
    this.routes.set(route, handler);
  }

  count(route: string): number {
    return this.calls.filter((c) => `${c.method} ${c.path}` === route).length;
  }

  api(): PokerMindApi {
    const fetchImpl = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = new URL(String(input), "http://ui.test");
      const path = url.pathname.replace(/^\/api/, "");
      const method = init?.method ?? "GET";
      const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
      const call = { method, path, body };
      this.calls.push(call);
      const handler = this.routes.get(`${method} ${path}`);
      if (!handler) return new Response(JSON.stringify({ detail: "no route" }), { status: 404 });
      const reply = await handler(call);
      if (reply === "network-error") throw new TypeError("fetch failed");
      if (reply.status === 204) return new Response(null, { status: 204 });
      return new Response(JSON.stringify(reply.body ?? null), {
        status: reply.status,
        headers: { "content-type": "application/json" },
      });
    };
    return new PokerMindApi("/api", fetchImpl as typeof fetch);
  }
}

export function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
