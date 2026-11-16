"use client";

import { useState } from "react";

import type { PokerMindApi } from "@/lib/api";
import { useGame } from "@/lib/useGame";

import { ActionBar } from "./ActionBar";
import { AiStatus } from "./AiStatus";
import { Dashboard } from "./Dashboard";
import { chips } from "./format";
import { Guide } from "./Guide";
import { History } from "./History";
import { Rules } from "./Rules";
import { Table } from "./Table";

function NewSessionButton({
  pending,
  inProgress,
  onConfirm,
}: {
  pending: boolean;
  inProgress: boolean;
  onConfirm: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  if (!confirming) {
    return (
      <button
        type="button"
        className="btn btn-ghost"
        disabled={pending}
        onClick={() => setConfirming(true)}
      >
        New session
      </button>
    );
  }
  return (
    <div className="confirm" role="group" aria-label="Confirm new session">
      <span className="small">
        {inProgress ? "Abandon this hand and" : "End this session and"} start over? The AI forgets
        what it learned about you.
      </span>
      <button
        type="button"
        className="btn btn-danger"
        disabled={pending}
        onClick={() => {
          setConfirming(false);
          onConfirm();
        }}
      >
        Start over
      </button>
      <button type="button" className="btn btn-ghost" onClick={() => setConfirming(false)}>
        Cancel
      </button>
    </div>
  );
}

type View = "play" | "learn";

function ViewTabs({ view, onChange }: { view: View; onChange: (v: View) => void }) {
  const tabs: [View, string][] = [
    ["play", "Play"],
    ["learn", "How PokerMind learns"],
  ];
  return (
    <div className="tabs" role="tablist" aria-label="View">
      {tabs.map(([id, label], i) => (
        <button
          key={id}
          type="button"
          role="tab"
          id={`tab-${id}`}
          aria-selected={view === id}
          aria-controls={`panel-${id}`}
          tabIndex={view === id ? 0 : -1}
          className={`tab${view === id ? " active" : ""}`}
          onClick={() => onChange(id)}
          onKeyDown={(e) => {
            if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
            e.preventDefault();
            const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
            if (next) {
              onChange(next[0]);
              document.getElementById(`tab-${next[0]}`)?.focus();
            }
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export function Game({ api }: { api?: PokerMindApi }) {
  const game = useGame(api);
  const { phase, session, history, metrics, strategy, pending, notice } = game;
  const hand = session?.hand ?? null;
  const offline = phase === "disconnected";
  const [view, setView] = useState<View>("play");
  const showView = (v: View) => {
    setView(v);
    if (v === "learn") void game.refresh();
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            ♠
          </span>
          <div>
            <h1>PokerMind</h1>
            <p className="tagline">Heads-up Leduc Hold&apos;em against an adaptive AI</p>
          </div>
        </div>
        {session && (
          <div className="session-bar">
            <span className="stat">
              Hands <strong>{session.stats.hands_played}</strong>
            </span>
            <span className="stat">
              Your total{" "}
              <strong
                className={
                  session.stats.human_net_chips > 0
                    ? "pos"
                    : session.stats.human_net_chips < 0
                      ? "neg"
                      : ""
                }
                data-testid="human-total"
              >
                {chips(session.stats.human_net_chips)}
              </strong>
            </span>
            <NewSessionButton
              pending={pending || offline}
              inProgress={hand?.status === "awaiting_human"}
              onConfirm={() => void game.startSession()}
            />
          </div>
        )}
      </header>

      {notice && phase !== "disconnected" && (
        <div className={`notice notice-${notice.kind}`} role="alert">
          <span>{notice.text}</span>
          <button type="button" className="btn btn-ghost btn-small" onClick={game.dismissNotice}>
            Dismiss
          </button>
        </div>
      )}

      {offline && (
        <div className="notice notice-error" role="alert">
          <span>
            Cannot reach the PokerMind server. Make sure <code>pokermind serve</code> is running,
            then retry. If the server was restarted, its in-memory sessions are gone.
          </span>
          <button
            type="button"
            className="btn btn-primary btn-small"
            disabled={pending}
            onClick={() => void game.retry()}
          >
            {pending ? "Retrying…" : "Retry"}
          </button>
        </div>
      )}

      {session && (phase === "playing" || offline) && (
        <ViewTabs view={view} onChange={showView} />
      )}

      {session && (phase === "playing" || offline) && view === "learn" ? (
        <main id="panel-learn" role="tabpanel" aria-labelledby="tab-learn" className="learn-view">
          <Dashboard
            metrics={metrics}
            strategy={strategy}
            pending={pending || offline}
            onRefresh={() => void game.refresh()}
          />
        </main>
      ) : (
      <main
        className="layout"
        {...(session && (phase === "playing" || offline)
          ? { id: "panel-play", role: "tabpanel", "aria-labelledby": "tab-play" }
          : {})}
      >
        <div className="main-col">
          {phase === "booting" && (
            <div className="empty" role="status">
              Connecting to PokerMind…
            </div>
          )}

          {(phase === "no_session" || phase === "lost") && (
            <div className="empty">
              {phase === "lost" ? (
                <>
                  <h2>This session no longer exists</h2>
                  <p>
                    The server keeps sessions in memory only. It was probably restarted, so the
                    game and what the AI learned about you are gone.
                  </p>
                </>
              ) : (
                <>
                  <h2>Play Leduc Hold&apos;em against PokerMind</h2>
                  <p>
                    PokerMind starts from a near-equilibrium strategy and adapts to how you play,
                    without ever becoming more exploitable than a fixed, verified limit.
                  </p>
                </>
              )}
              <button
                type="button"
                className="btn btn-primary btn-wide"
                disabled={pending}
                onClick={() => void game.startSession()}
              >
                {pending ? "Starting…" : "Start a new session"}
              </button>
            </div>
          )}

          {session && (phase === "playing" || offline) && (
            <>
              <Guide />
              {hand ? (
                <Table hand={hand} />
              ) : (
                <div className="empty table-empty">
                  <p>Session ready. Deal the first hand when you are.</p>
                </div>
              )}
              <ActionBar
                hand={hand}
                pending={pending}
                disabled={offline}
                onAct={(a) => void game.act(a)}
                onDeal={() => void game.deal()}
              />
            </>
          )}
          <Rules />
        </div>

        {session && (
          <aside className="side-col">
            <AiStatus
              strategy={session.strategy}
              metrics={metrics}
              onOpenDashboard={() => showView("learn")}
            />
            <History hands={history} total={session.hands_played} />
          </aside>
        )}
      </main>
      )}

      <footer className="footer muted small">
        Local session · nothing is stored on disk · the server forgets sessions when it restarts
      </footer>
    </div>
  );
}
