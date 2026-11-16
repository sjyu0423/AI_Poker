/** A scripted stand-in for the FastAPI backend, injected through PokerMindApi's fetch. */

import { PokerMindApi } from "@/lib/api";
import type { HandView, SessionMetrics, SessionState } from "@/lib/types";

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

export function metrics(handsObserved = 0, running = false): SessionMetrics {
  return {
    session_id: "s1",
    hands_played: handsObserved,
    verified: {
      active_strategy: {
        version: 1,
        source: "baseline_cfr",
        certified_exploitability: 0.0029968965501324385,
        exploitability_limit: 0.10299689655013244,
        based_on_hands: 0,
        installed_after_hand: 0,
      },
      exploitability_limit: 0.10299689655013244,
      observation_counts: {
        hands_observed: handsObserved,
        hands_in_model: handsObserved,
        human_decisions_in_model: 0,
        showdowns_in_model: 0,
        change_resets_after_hands: [],
      },
    },
    strategy_updates: {
      running,
      pending: false,
      recompute_every_hands: 50,
      next_scheduled_after_hands: 50,
      last_success: null,
      last_failure: null,
    },
  };
}

export class FakeServer {
  calls: Call[] = [];
  routes = new Map<string, Handler>();
  current: SessionState = state();
  metrics: SessionMetrics = metrics();
  historyHands: HandView[] = [];

  constructor() {
    this.on("GET /health", () => ({ status: 200, body: { status: "ok", api_version: "1" } }));
    this.on("POST /sessions", () => ({ status: 201, body: this.current }));
    this.on("GET /sessions/s1", () => ({ status: 200, body: this.current }));
    this.on("DELETE /sessions/s1", () => ({ status: 204 }));
    this.on("GET /sessions/s1/metrics", () => ({ status: 200, body: this.metrics }));
    this.on("GET /sessions/s1/history", () => ({
      status: 200,
      body: { session_id: "s1", total: this.historyHands.length, offset: 0, hands: this.historyHands },
    }));
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
