/**
 * Typed client for the PokerMind API. Requests go to `/api/*`, which the Next.js
 * server forwards to the FastAPI backend.
 *
 * Errors are classified, never swallowed:
 * - `ApiError` carries the HTTP status and the backend's `detail`
 *   (400 illegal action, 404 unknown session, 409 stale or conflicting state,
 *   422 malformed request);
 * - `ConnectionError` is raised when the backend cannot be reached, including a
 *   gateway failure of the forwarding server (502/503/504).
 */

import type {
  ActionName,
  HistoryPage,
  ServerConfig,
  SessionMetrics,
  SessionState,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

export class ConnectionError extends Error {
  constructor(message = "The PokerMind server cannot be reached.") {
    super(message);
    this.name = "ConnectionError";
  }
}

type Fetch = typeof fetch;

const GATEWAY_FAILURES = new Set([500, 502, 503, 504]);

function detailOf(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI 422: a list of validation errors
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
        .join("; ");
    }
  }
  return `Request failed (HTTP ${status}).`;
}

export class PokerMindApi {
  constructor(
    private readonly base = "/api",
    private readonly fetchImpl: Fetch = (...args) => fetch(...args),
  ) {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.base}${path}`, {
        method,
        headers: body === undefined ? undefined : { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
      });
    } catch {
      throw new ConnectionError();
    }
    if (response.status === 204) return undefined as T;
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      // The forwarding server answers 5xx without a FastAPI body when the backend is down.
      const fromBackend = payload !== null && typeof payload === "object" && "detail" in payload;
      if (GATEWAY_FAILURES.has(response.status) && !fromBackend) throw new ConnectionError();
      throw new ApiError(response.status, detailOf(payload, response.status));
    }
    return payload as T;
  }

  health(): Promise<{ status: string; api_version: string }> {
    return this.request("GET", "/health");
  }

  config(): Promise<ServerConfig> {
    return this.request("GET", "/config");
  }

  createSession(): Promise<SessionState> {
    return this.request("POST", "/sessions", {});
  }

  getSession(id: string): Promise<SessionState> {
    return this.request("GET", `/sessions/${encodeURIComponent(id)}`);
  }

  deleteSession(id: string): Promise<void> {
    return this.request("DELETE", `/sessions/${encodeURIComponent(id)}`);
  }

  deal(id: string): Promise<SessionState> {
    return this.request("POST", `/sessions/${encodeURIComponent(id)}/hands`);
  }

  act(id: string, action: ActionName, stateVersion: number): Promise<SessionState> {
    return this.request("POST", `/sessions/${encodeURIComponent(id)}/actions`, {
      action,
      state_version: stateVersion,
    });
  }

  history(id: string, offset: number, limit: number): Promise<HistoryPage> {
    const q = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    return this.request("GET", `/sessions/${encodeURIComponent(id)}/history?${q}`);
  }

  metrics(id: string): Promise<SessionMetrics> {
    return this.request("GET", `/sessions/${encodeURIComponent(id)}/metrics`);
  }
}
