import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { Game } from "@/components/Game";
import { SESSION_KEY } from "@/lib/useGame";

import { deferred, FakeServer, hand, metrics, state } from "./fake";

let server: FakeServer;

beforeEach(() => {
  server = new FakeServer();
});

async function startPlaying(): Promise<ReturnType<typeof userEvent.setup>> {
  const user = userEvent.setup();
  render(<Game api={server.api()} />);
  await user.click(await screen.findByRole("button", { name: /start a new session/i }));
  await screen.findByRole("button", { name: /deal first hand/i });
  return user;
}

describe("gameplay", () => {
  it("starts a session, plays a full hand and starts the next one", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => {
      server.metrics = metrics(1);
      return {
        status: 200,
        body: (server.current = state({
          state_version: 2,
          hands_played: 1,
          hand: hand({
            status: "complete",
            public_card: "Q0",
            round: 2,
            to_act: null,
            legal_actions: [],
            actions: [
              { actor: "human", round: 1, action: "call", label: "check" },
              { actor: "ai", round: 1, action: "call", label: "check" },
              { actor: "human", round: 2, action: "call", label: "check" },
              { actor: "ai", round: 2, action: "call", label: "check" },
            ],
            result: {
              ended_by: "showdown",
              winner: "human",
              human_net: 1,
              ai_net: -1,
              pot: 2,
              ai_card: "J0",
            },
          }),
        })),
      };
    });
    const user = await startPlaying();
    expect(window.sessionStorage.getItem(SESSION_KEY)).toBe("s1");
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));

    expect(await screen.findByRole("img", { name: "King of hearts" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /PokerMind's private card \(face down\)/ })).toBeVisible();

    await user.click(screen.getByRole("button", { name: /^check/i }));
    const sent = server.calls.find((c) => c.path === "/sessions/s1/actions");
    expect(sent?.body).toEqual({ action: "call", state_version: 1 });

    const result = await screen.findByTestId("result");
    expect(result).toHaveTextContent("You win at showdown");
    expect(result).toHaveTextContent("You won 1 chip (+1)");
    expect(screen.getByRole("img", { name: "Jack of spades" })).toBeInTheDocument();
    expect(screen.getByTestId("human-total")).toHaveTextContent("0"); // from server stats
    expect(screen.getByRole("button", { name: /next hand/i })).toBeEnabled();
  });

  it("renders exactly the legal actions the server lists", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({
        state_version: 1,
        hand: hand({
          human_seat: 1,
          first_to_act: "ai",
          actions: [{ actor: "ai", round: 1, action: "raise", label: "bet" }],
          contributions: { human: 1, ai: 3 },
          pot: 4,
          legal_actions: [
            { action: "fold", label: "fold" },
            { action: "call", label: "call" },
            { action: "raise", label: "raise" },
          ],
        }),
      })),
    }));
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    const group = await screen.findByRole("group", { name: "Your actions" });
    const names = within(group)
      .getAllByRole("button")
      .map((b) => b.textContent?.replace(/[A-Z]$/, "").trim());
    expect(names).toEqual(["Fold", "Call", "Raise"]);
    expect(screen.getByTestId("pot")).toHaveTextContent("4");
    expect(screen.queryByRole("button", { name: /^check/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^bet/i })).toBeNull();
  });

  it("shows the public card only once the server reveals it", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => ({
      status: 200,
      body: (server.current = state({
        state_version: 2,
        hand: hand({ round: 2, public_card: "Q1", actions: [] }),
      })),
    }));
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    expect(await screen.findByRole("img", { name: /public card: dealt after round 1/i })).toBeVisible();
    expect(screen.queryByRole("img", { name: "Queen of hearts" })).toBeNull();
    await user.click(screen.getByRole("button", { name: /^check/i }));
    expect(await screen.findByRole("img", { name: "Queen of hearts" })).toBeVisible();
    expect(screen.queryByRole("img", { name: /dealt after round 1/i })).toBeNull();
  });

  it("never shows the AI's card unless the server sends it", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => ({
      status: 200,
      body: (server.current = state({
        state_version: 2,
        hands_played: 1,
        hand: hand({
          status: "complete",
          to_act: null,
          legal_actions: [],
          actions: [
            { actor: "human", round: 1, action: "raise", label: "bet" },
            { actor: "ai", round: 1, action: "fold", label: "fold" },
          ],
          result: { ended_by: "fold", winner: "human", human_net: 1, ai_net: -1, pot: 4, ai_card: null },
        }),
      })),
    }));
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    const faces = () =>
      Array.from(document.querySelectorAll("[data-card]")).map((el) => el.getAttribute("data-card"));
    await screen.findByRole("img", { name: "King of hearts" });
    expect(faces()).toEqual(["K1"]); // only the human's own card is face up
    await user.click(screen.getByRole("button", { name: /^bet/i }));
    await screen.findByTestId("result");
    expect(faces()).toEqual(["K1"]);
    expect(screen.getByText(/card stays hidden after a fold/i)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /face down/i })).toBeInTheDocument();
  });
});

describe("request handling", () => {
  it("prevents duplicate submissions while a request is pending", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    const gate = deferred<{ status: number; body: unknown }>();
    server.on("POST /sessions/s1/actions", () => gate.promise);
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    const check = await screen.findByRole("button", { name: /^check/i });
    await user.click(check);
    expect(check).toBeDisabled();
    expect(screen.getByRole("button", { name: /^bet/i })).toBeDisabled();
    await user.click(check);
    await user.keyboard("kb");
    expect(server.count("POST /sessions/s1/actions")).toBe(1);
    await act(async () => {
      gate.resolve({ status: 200, body: (server.current = state({ state_version: 2, hand: hand() })) });
    });
    await waitFor(() => expect(screen.getByRole("button", { name: /^check/i })).toBeEnabled());
  });

  it("recovers from a stale state_version (409) by refetching", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => {
      // another tab acted first: the server is ahead
      server.current = state({ state_version: 3, hand: hand({ hand_number: 2, human_card: "J0" }) });
      return { status: 409, body: { detail: "stale state_version 1; current is 3" } };
    });
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    await user.click(await screen.findByRole("button", { name: /^check/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/moved on/i);
    expect(await screen.findByRole("img", { name: "Jack of spades" })).toBeInTheDocument();
    expect(server.count("GET /sessions/s1")).toBeGreaterThanOrEqual(1);
    // the next action uses the refreshed version
    server.on("POST /sessions/s1/actions", () => ({ status: 200, body: server.current }));
    await user.click(screen.getByRole("button", { name: /^check/i }));
    const last = server.calls.filter((c) => c.path === "/sessions/s1/actions").at(-1);
    expect(last?.body).toEqual({ action: "call", state_version: 3 });
  });

  it("shows a rejected action without advancing the game (400)", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => ({
      status: 400,
      body: { detail: "'raise' is not legal now; legal actions: ['call']" },
    }));
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    await user.click(await screen.findByRole("button", { name: /^bet/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("is not legal now");
    expect(screen.queryByTestId("result")).toBeNull();
    expect(screen.getByRole("button", { name: /^check/i })).toBeEnabled();
  });

  it("explains a lost session (404) after a server restart", async () => {
    window.sessionStorage.setItem(SESSION_KEY, "s1");
    server.on("GET /sessions/s1", () => ({ status: 404, body: { detail: "unknown session" } }));
    render(<Game api={server.api()} />);
    expect(await screen.findByText(/this session no longer exists/i)).toBeInTheDocument();
    expect(screen.getByText(/keeps sessions in memory/i)).toBeInTheDocument();
    expect(window.sessionStorage.getItem(SESSION_KEY)).toBeNull();
    expect(screen.getByRole("button", { name: /start a new session/i })).toBeEnabled();
  });

  it("enters a disconnected state and retries", async () => {
    server.on("POST /sessions/s1/hands", () => "network-error");
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /deal first hand/i }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/cannot reach the pokermind server/i);
    expect(screen.getByRole("button", { name: /deal first hand/i })).toBeDisabled();
    server.current = state({ state_version: 1, hand: hand() });
    await user.click(within(alert).getByRole("button", { name: /retry/i }));
    expect(await screen.findByRole("img", { name: "King of hearts" })).toBeInTheDocument();
    expect(screen.queryByText(/cannot reach/i)).toBeNull();
    expect(server.count("GET /health")).toBe(1);
  });

  it("resumes the tab's session after a reload", async () => {
    window.sessionStorage.setItem(SESSION_KEY, "s1");
    server.current = state({ state_version: 4, hand: hand({ hand_number: 3 }) });
    render(<Game api={server.api()} />);
    expect(await screen.findByRole("region", { name: "Hand 3" })).toBeInTheDocument();
    expect(server.count("POST /sessions")).toBe(0);
  });

  it("asks before abandoning a session and creates a fresh one", async () => {
    const user = await startPlaying();
    await user.click(screen.getByRole("button", { name: /new session/i }));
    expect(screen.getByRole("group", { name: /confirm new session/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /start over/i }));
    await waitFor(() => expect(server.count("DELETE /sessions/s1")).toBe(1));
    expect(server.count("POST /sessions")).toBe(2);
  });
});

describe("AI status", () => {
  it("shows verified facts and a Learning state during recomputation", async () => {
    server.current = state({
      strategy: {
        active_version: 2,
        source: "rnr",
        certified_exploitability: 0.0817929,
        exploitability_limit: 0.1029969,
        recomputation_running: true,
      },
    });
    server.metrics = metrics(25, true);
    await startPlaying();
    expect(await screen.findByText("Learning…")).toBeInTheDocument();
    expect(screen.getByTestId("strategy-version")).toHaveTextContent("v2");
    expect(screen.getByText("0.0818 chips/hand")).toBeInTheDocument();
    expect(screen.getByText("0.1030 chips/hand")).toBeInTheDocument();
    expect(await screen.findByText("25")).toBeInTheDocument();
    expect(screen.getByText(/every 50 hands/i)).toBeInTheDocument();
  });

  it("does not show Learning when no update is running", async () => {
    await startPlaying();
    await screen.findByTestId("strategy-version");
    expect(screen.queryByText("Learning…")).toBeNull();
  });
});

describe("keyboard", () => {
  it("plays with shortcuts and reaches every control with Tab", async () => {
    server.on("POST /sessions/s1/hands", () => ({
      status: 201,
      body: (server.current = state({ state_version: 1, hand: hand() })),
    }));
    server.on("POST /sessions/s1/actions", () => ({
      status: 200,
      body: (server.current = state({ state_version: 2, hand: hand() })),
    }));
    const user = await startPlaying();
    await user.keyboard("n");
    await screen.findByRole("img", { name: "King of hearts" });
    expect(server.count("POST /sessions/s1/hands")).toBe(1);
    await user.keyboard("b");
    await waitFor(() => expect(server.count("POST /sessions/s1/actions")).toBe(1));
    expect(server.calls.find((c) => c.path.endsWith("/actions"))?.body).toEqual({
      action: "raise",
      state_version: 1,
    });
    await user.keyboard("f"); // fold is not legal here: nothing is sent
    expect(server.count("POST /sessions/s1/actions")).toBe(1);
    const check = screen.getByRole("button", { name: /^check/i });
    expect(check).toHaveAttribute("aria-keyshortcuts", "K");
    let reached = false;
    for (let i = 0; i < 12 && !reached; i++) {
      await user.tab();
      reached = document.activeElement === check;
    }
    expect(reached).toBe(true);
    await user.keyboard("{Enter}");
    await waitFor(() => expect(server.count("POST /sessions/s1/actions")).toBe(2));
  });
});
