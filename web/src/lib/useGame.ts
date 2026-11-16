"use client";

/**
 * Game controller: the only place that talks to the API.
 *
 * - The server's `SessionState` is the single source of truth. Nothing about the
 *   game (legal actions, cards, payouts) is computed here.
 * - One request at a time: `busy` is a ref, checked synchronously, so a double click
 *   or a repeated key press cannot submit twice. Controls are also disabled while
 *   `pending` is true.
 * - Every action carries the `state_version` the player saw. A response is applied
 *   only if it is not older than the state already shown.
 * - Errors never advance the game:
 *   - 409 refetches the current state;
 *   - 400 and 422 show the reason and refetch;
 *   - 404 means the session no longer exists (the server keeps sessions in memory);
 *   - a connection failure switches to the disconnected state with a retry.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, ConnectionError, PokerMindApi } from "./api";
import type { ActionName, HandView, SessionMetrics, SessionState } from "./types";

export type Phase = "booting" | "no_session" | "playing" | "lost" | "disconnected";

export interface Notice {
  kind: "info" | "error";
  text: string;
}

export const SESSION_KEY = "pokermind.session";
export const HISTORY_SIZE = 8;
export const POLL_MS = 2500;

export interface Game {
  phase: Phase;
  session: SessionState | null;
  history: HandView[];
  metrics: SessionMetrics | null;
  pending: boolean;
  notice: Notice | null;
  dismissNotice: () => void;
  startSession: () => Promise<void>;
  deal: () => Promise<void>;
  act: (action: ActionName) => Promise<void>;
  retry: () => Promise<void>;
}

function describe(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return "Unexpected error.";
}

export function useGame(api?: PokerMindApi): Game {
  const client = useMemo(() => api ?? new PokerMindApi(), [api]);
  const [phase, setPhase] = useState<Phase>("booting");
  const [session, setSession] = useState<SessionState | null>(null);
  const [history, setHistory] = useState<HandView[]>([]);
  const [metrics, setMetrics] = useState<SessionMetrics | null>(null);
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const busy = useRef(false);
  const current = useRef<SessionState | null>(null);

  /** Show a server state unless an already shown one is newer. */
  const apply = useCallback((next: SessionState) => {
    const shown = current.current;
    if (
      shown &&
      shown.session_id === next.session_id &&
      next.state_version < shown.state_version
    ) {
      return;
    }
    current.current = next;
    setSession(next);
    setPhase("playing");
  }, []);

  const forget = useCallback(() => {
    current.current = null;
    setSession(null);
    setHistory([]);
    setMetrics(null);
    window.sessionStorage.removeItem(SESSION_KEY);
  }, []);

  /** History and metrics: secondary views, refreshed after hands and while learning. */
  const refreshSide = useCallback(
    async (id: string, handsPlayed: number) => {
      const offset = Math.max(0, handsPlayed - HISTORY_SIZE);
      const [page, m] = await Promise.all([
        client.history(id, offset, HISTORY_SIZE),
        client.metrics(id),
      ]);
      if (current.current?.session_id !== id) return;
      setHistory([...page.hands].reverse());
      setMetrics(m);
    },
    [client],
  );

  const handleError = useCallback(
    async (err: unknown) => {
      if (err instanceof ConnectionError) {
        setPhase("disconnected");
        setNotice({ kind: "error", text: err.message });
        return;
      }
      if (err instanceof ApiError && err.status === 404) {
        forget();
        setPhase("lost");
        setNotice(null);
        return;
      }
      const id = current.current?.session_id;
      if (err instanceof ApiError && err.status === 409) {
        setNotice({
          kind: "info",
          text: "The game had already moved on. Showing the current state.",
        });
      } else {
        setNotice({ kind: "error", text: describe(err) });
      }
      if (id && err instanceof ApiError) {
        try {
          const fresh = await client.getSession(id);
          current.current = fresh; // authoritative resync, even if not newer
          setSession(fresh);
          setPhase("playing");
          await refreshSide(id, fresh.hands_played);
        } catch (inner) {
          if (inner instanceof ConnectionError) setPhase("disconnected");
          else if (inner instanceof ApiError && inner.status === 404) {
            forget();
            setPhase("lost");
          }
        }
      }
    },
    [client, forget, refreshSide],
  );

  /** Run one request; refuse to start another while one is in flight. */
  const run = useCallback(
    async (op: () => Promise<void>) => {
      if (busy.current) return;
      busy.current = true;
      setPending(true);
      try {
        await op();
      } catch (err) {
        await handleError(err);
      } finally {
        busy.current = false;
        setPending(false);
      }
    },
    [handleError],
  );

  // Resume the tab's session after a reload, if the server still has it.
  // No controls are shown while booting, so this needs no request lock.
  useEffect(() => {
    let cancelled = false;
    const id = window.sessionStorage.getItem(SESSION_KEY);
    const boot = id ? client.getSession(id) : Promise.resolve(null);
    boot
      .then(async (state) => {
        if (cancelled) return;
        if (state === null) {
          setPhase("no_session");
          return;
        }
        apply(state);
        await refreshSide(state.session_id, state.hands_played);
      })
      .catch((err: unknown) => {
        if (!cancelled) void handleError(err);
      });
    return () => {
      cancelled = true;
    };
  }, [client, apply, refreshSide, handleError]);

  const startSession = useCallback(
    () =>
      run(async () => {
        const old = current.current?.session_id;
        if (old) {
          try {
            await client.deleteSession(old);
          } catch (err) {
            if (!(err instanceof ApiError && err.status === 404)) throw err;
          }
        }
        forget();
        const state = await client.createSession();
        window.sessionStorage.setItem(SESSION_KEY, state.session_id);
        setNotice(null);
        apply(state);
        await refreshSide(state.session_id, state.hands_played);
      }),
    [client, apply, forget, refreshSide, run],
  );

  const deal = useCallback(
    () =>
      run(async () => {
        const id = current.current?.session_id;
        if (!id) return;
        setNotice(null);
        apply(await client.deal(id));
      }),
    [client, apply, run],
  );

  const act = useCallback(
    (action: ActionName) =>
      run(async () => {
        const shown = current.current;
        if (!shown?.hand || shown.hand.status !== "awaiting_human") return;
        setNotice(null);
        const next = await client.act(shown.session_id, action, shown.state_version);
        apply(next);
        if (next.hand?.status === "complete") {
          await refreshSide(next.session_id, next.hands_played);
        }
      }),
    [client, apply, refreshSide, run],
  );

  const retry = useCallback(
    () =>
      run(async () => {
        await client.health();
        const id = current.current?.session_id ?? window.sessionStorage.getItem(SESSION_KEY);
        setNotice(null);
        if (!id) {
          setPhase("no_session");
          return;
        }
        const state = await client.getSession(id);
        current.current = state;
        setSession(state);
        setPhase("playing");
        await refreshSide(id, state.hands_played);
      }),
    [client, refreshSide, run],
  );

  // While the AI recomputes in the background, poll its status (never during a request).
  const learning =
    session?.strategy.recomputation_running === true ||
    metrics?.strategy_updates.running === true ||
    metrics?.strategy_updates.pending === true;
  useEffect(() => {
    if (phase !== "playing" || !learning) return;
    const timer = window.setInterval(() => {
      const shown = current.current;
      if (busy.current || !shown) return;
      void (async () => {
        try {
          const state = await client.getSession(shown.session_id);
          if (busy.current) return;
          apply(state);
          await refreshSide(state.session_id, state.hands_played);
        } catch (err) {
          if (err instanceof ConnectionError) setPhase("disconnected");
          else if (err instanceof ApiError && err.status === 404) {
            forget();
            setPhase("lost");
          }
        }
      })();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [phase, learning, client, apply, refreshSide, forget]);

  return {
    phase,
    session,
    history,
    metrics,
    pending,
    notice,
    dismissNotice: () => setNotice(null),
    startSession,
    deal,
    act,
    retry,
  };
}
