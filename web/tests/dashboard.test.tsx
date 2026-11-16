import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { inPlayFrom } from "@/components/Dashboard";
import { Game } from "@/components/Game";
import { Table } from "@/components/Table";
import { SESSION_KEY } from "@/lib/useGame";

import { FakeServer, hand, metrics, state, updates, version } from "./fake";

let server: FakeServer;

beforeEach(() => {
  server = new FakeServer();
});

async function openDashboard(): Promise<ReturnType<typeof userEvent.setup>> {
  const user = userEvent.setup();
  render(<Game api={server.api()} />);
  await user.click(await screen.findByRole("button", { name: /start a new session/i }));
  await user.click(await screen.findByRole("tab", { name: /how pokermind learns/i }));
  await screen.findByRole("tabpanel");
  await screen.findByTestId("dash-hands");
  return user;
}

describe("learning dashboard", () => {
  it("shows the empty state, every situation as sparse, and the disclaimer", async () => {
    await openDashboard();
    expect(screen.getByTestId("dash-hands")).toHaveTextContent("0");
    expect(screen.getByText(/no decisions observed yet/i)).toBeInTheDocument();
    const cards = screen.getAllByRole("article");
    expect(cards).toHaveLength(6);
    for (const card of cards) {
      expect(within(card).getByText(/0 decisions · sparse/)).toBeInTheDocument();
      expect(within(card).getByText(/hasn't seen you in this situation yet/)).toBeInTheDocument();
    }
    expect(screen.getByRole("note")).toHaveTextContent(
      "have not been validated for real human opponents",
    );
    expect(screen.getByText(/the prior counts as 64 decisions/i)).toBeInTheDocument();
  });

  it("separates the population prior from the personal estimate", async () => {
    server.metrics = metrics(20, false, { observed: { "R1:fcr": { raise: 6, call: 2 } } });
    await openDashboard();
    const card = screen.getByRole("article", { name: "Round 1 · facing a bet" });
    expect(within(card).getByText("8 decisions")).toBeInTheDocument();
    const raise = within(card).getByRole("listitem", { name: /^raise:/ });
    // (6 + 64·0.18) / (8 + 64) = 0.243
    expect(raise).toHaveAccessibleName(
      "raise: estimated 24%, 90% interval 15% to 33%, typical player 18%, seen 6 of 8",
    );
    expect(
      within(card).getByText(
        "You've raised 6 of 8 times here (75%), more often than a typical player (18%).",
      ),
    ).toBeInTheDocument();
    const observedBadges = within(card).getAllByText("Observed");
    const estimateBadges = within(card).getAllByText("Estimate");
    expect(observedBadges.length).toBeGreaterThan(0);
    expect(estimateBadges.length).toBeGreaterThan(0);
    // other situations stay sparse
    expect(screen.getByRole("article", { name: "Round 2 · facing a bet" })).toHaveTextContent(
      "sparse",
    );
  });

  it("shows certified strategy facts and the version history", async () => {
    server.versions = [
      version(),
      version({
        version: 2,
        source: "rnr",
        rnr_p: 0.5,
        certified_exploitability: 0.0662,
        based_on_hands: 50,
        installed_after_hand: 50,
      }),
    ];
    server.metrics = metrics(50, false, {
      active: server.versions[1],
      updates: {
        last_success: {
          job_id: 1,
          version: 2,
          reason: "scheduled",
          based_on_hands: 50,
          seconds: 7.4,
          finished_at: 0,
        },
      },
    });
    await openDashboard();
    expect(screen.getByTestId("dash-version")).toHaveTextContent("v2");
    expect(screen.getByTestId("dash-exploitability")).toHaveTextContent("0.0662 chips/hand");
    expect(screen.getByText("v2 from 50 hands (7.4 s)")).toBeInTheDocument();
    const meter = screen.getByRole("meter", { name: /certified exploitability/i });
    expect(meter).toHaveAttribute("aria-valuetext", "0.0662 of at most 0.1030 chips per hand");
    const rows = within(screen.getByRole("table", { name: "" }).closest("table")!)
      .getAllByRole("row")
      .slice(1);
    expect(rows.map((r) => r.textContent)).toEqual([
      "v2Adapted to you (RNR) p=0.50.06625051",
      "v1Base CFR strategy0.003001",
    ]);
    expect(
      screen.getByText(/Strategy v2 is adapted to your play \(from 50 hands\) and passed the exploitability check/),
    ).toBeInTheDocument();
  });

  it("shows a running update and then the new version after refresh", async () => {
    server.current = state({
      strategy: {
        active_version: 1,
        source: "baseline_cfr",
        certified_exploitability: 0.003,
        exploitability_limit: 0.103,
        recomputation_running: true,
      },
    });
    server.metrics = metrics(50, true, {
      updates: {
        running_job: { job_id: 1, reason: "scheduled", based_on_hands: 50, elapsed_seconds: 1 },
      },
    });
    const user = await openDashboard();
    expect(screen.getByTestId("dash-update-status")).toHaveTextContent("Preparing…");
    expect(screen.getByText(/preparing an updated strategy from 50 hands/i)).toBeInTheDocument();
    expect(screen.getByTestId("dash-version")).toHaveTextContent("v1");
    // the update finishes
    server.versions = [version(), version({ version: 2, source: "cfr_fallback", based_on_hands: 50, installed_after_hand: 50 })];
    server.metrics = metrics(50, false, { active: server.versions[1], updates: updates() });
    server.current = state();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(screen.getByTestId("dash-version")).toHaveTextContent("v2"));
    expect(screen.getByTestId("dash-update-status")).toHaveTextContent("Idle");
    expect(screen.getByText(/not confident that adapting would pay off/)).toBeInTheDocument();
  });

  it("explains a lost session from the dashboard", async () => {
    const user = await openDashboard();
    server.on("GET /sessions/s1", () => ({ status: 404, body: { detail: "unknown session" } }));
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText(/this session no longer exists/i)).toBeInTheDocument();
    expect(window.sessionStorage.getItem(SESSION_KEY)).toBeNull();
  });

  it("switches views with the keyboard and keeps the game", async () => {
    const user = await openDashboard();
    const learn = screen.getByRole("tab", { name: /how pokermind learns/i });
    expect(learn).toHaveAttribute("aria-selected", "true");
    learn.focus();
    await user.keyboard("{ArrowLeft}");
    const play = screen.getByRole("tab", { name: "Play" });
    expect(play).toHaveAttribute("aria-selected", "true");
    expect(play).toHaveFocus();
    expect(await screen.findByRole("button", { name: /deal first hand/i })).toBeInTheDocument();
    // N deals even while a tab button has focus
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    await user.keyboard("n");
    await waitFor(() => expect(server.count("POST /sessions/s1/hands")).toBe(1));
  });
});

describe("version history", () => {
  it("marks a version replaced before any hand was dealt with it", () => {
    const v1 = version();
    const v2 = version({ version: 2, source: "cfr_fallback", based_on_hands: 10, installed_after_hand: 25 });
    const v3 = version({ version: 3, source: "rnr", based_on_hands: 25, installed_after_hand: 25 });
    const all = [v1, v2, v3];
    expect(inPlayFrom(v1, all)).toBe("1");
    expect(inPlayFrom(v2, all)).toBe("replaced before use");
    expect(inPlayFrom(v3, all)).toBe("26");
  });
});

describe("hand log", () => {
  it("uses the server's table labels", () => {
    render(
      <Table
        hand={hand({
          actions: [
            { actor: "human", round: 1, action: "call", label: "check" },
            { actor: "ai", round: 1, action: "raise", label: "bet" },
            { actor: "human", round: 1, action: "raise", label: "raise" },
            { actor: "ai", round: 1, action: "call", label: "call" },
            { actor: "human", round: 2, action: "raise", label: "bet" },
            { actor: "ai", round: 2, action: "fold", label: "fold" },
          ],
        })}
      />,
    );
    const log = screen.getByRole("list", { name: "Actions this hand" });
    const items = within(log).getAllByRole("listitem").slice(1).map((li) => li.textContent);
    expect(items).toEqual([
      "R1 You check",
      "R1 PokerMind bets",
      "R1 You raise",
      "R1 PokerMind calls",
      "R2 You bet",
      "R2 PokerMind folds",
    ]);
  });
});
