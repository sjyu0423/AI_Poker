# PokerMind web (M8B, M8C)

A local web interface for playing heads-up Leduc Hold'em against PokerMind's
adaptive AI. It is a Next.js 16 / React 19 / TypeScript app. It is a pure
client of the FastAPI backend (`pokermind serve`, contract in
`../docs/design/app_backend.md`), and it holds no game logic.

## Run

```bash
# terminal 1: the backend (from the repository root, needs pip install -e ".[app]")
pokermind serve                      # http://127.0.0.1:8000

# terminal 2: the web app
cd web
npm install
npm run build && npm run start       # http://localhost:3000  (or: npm run dev)
```

The browser talks only to the Next.js server, which forwards `/api/*` to the
backend, so the backend needs no CORS settings. To use another backend
address, set it **before building**:
`POKERMIND_API_URL=http://127.0.0.1:8001 npm run build`.

## Check

```bash
npm run typecheck   # tsc --noEmit (strict)
npm run lint        # ESLint (next/core-web-vitals + TypeScript)
npm test            # Vitest + Testing Library (jsdom, scripted fake backend)
npm run build       # production build
```

Usually you don't need these steps: `pokermind launch` (repository root)
checks the setup, builds when needed, and runs both services.

## Screenshots

`node scripts/capture-screenshots.mjs` drives the installed Google Chrome
(headless, real mouse and keyboard input) through a real session against a
running app (`pokermind launch --demo`). It writes PNGs to
`docs/screenshots/`. It needs Node.js ≥ 22 and has no npm dependencies.

## Structure

```
src/lib/types.ts        response types, mirroring the M8A views
src/lib/api.ts          typed client; ApiError (HTTP status + detail), ConnectionError
src/lib/useGame.ts      controller: one request at a time, state_version, error recovery
src/components/Game.tsx screen states: connecting, no session, playing, lost, disconnected
src/components/Table.tsx  cards, public card, round, pot, contributions, action log, result
src/components/ActionBar.tsx  buttons for exactly the server's legal actions; shortcuts
src/components/AiStatus.tsx   strategy version, learning state, hands observed, exploitability
src/components/History.tsx    last 8 completed hands (GET /history)
src/components/Dashboard.tsx  M8C learning dashboard: observations, tendencies, strategy evolution
src/lib/explain.ts      plain-language statements, each tagged observed/estimate/certified/status
src/components/Rules.tsx, Guide.tsx  rules and first-time guidance
tests/                  api, game, dashboard, explain tests; fake.ts (scripted backend)
```

## Behaviour

* **The server state is the single source of truth.** The UI shows whatever
  the server sends: the human's card, the public card when dealt, legal
  actions with their labels (check/bet/call/raise/fold), pot,
  contributions, results, and the AI's card only when a result carries it.
* **Every action carries the `state_version` the player saw.** Controls are
  disabled while a request is in flight, and a synchronous lock drops
  double clicks and repeated keys.
* **Errors never advance the game:**

  | error | what the UI does |
  |---|---|
  | 409 (stale or conflicting state) | notice, then refetches the current state |
  | 400 or 422 | shows the reason, then refetches |
  | 404 | the session is gone; explains that sessions are in memory and lost on a server restart |
  | backend unreachable | disconnected banner with Retry; controls disabled |

* **AI status** polls every 2.5 s, but only while a recomputation is running
  or pending, and shows "Learning…" meanwhile. A new strategy appears as a
  new version number.
* **Keyboard:** K check, B bet, C call, R raise, F fold, N next hand, and
  Tab/Enter on every button. Shortcuts are listed in `aria-keyshortcuts`.
* **Sessions:** the tab remembers its session id (`sessionStorage`) and
  resumes it after a reload. "New session" asks for confirmation, then
  deletes the old session.

## Learning dashboard (M8C)

The "How PokerMind learns" tab reads only `/metrics` and `/strategy`.

* **What PokerMind has observed** (counted). Shows hands observed, hands in
  the current model, the player's decisions, and cards seen at showdown.
* **Estimated tendencies** (estimates, with the disclaimer that intervals
  were not validated for human opponents). For each of the six betting
  situations it shows:
  * the posterior mean and 90 % interval;
  * the population prior, as a tick;
  * observed counts;
  * a *sparse* badge below 5 decisions. The prior counts as 64
    pseudo-decisions, so interval width alone would understate how little
    is known.
* **Strategy evolution** (certified). Shows:
  * the current version, base CFR or adapted;
  * certified exploitability against the limit, as a meter;
  * the last successful update and the background status;
  * the session's version history.

  The AI's own action probabilities are deliberately **not** shown, because
  they would reveal how to beat it.
* **Explanations** come only from served values. Each is tagged *Observed*,
  *Estimate*, *Certified* or *Status*. Adaptation is claimed only for an
  installed, certified RNR version, never because an estimate moved.
  Tendency statements need at least 5 decisions and a difference of at
  least 15 points from the typical player.

## Known limitations

* **Strategy history** lasts only for the session. The server keeps it in
  memory with the session.
* **Tendencies use the model's coarse situations** (round × facing a bet or
  not). The finer card-conditional reconstruction behind each strategy
  update is not exposed.
* **Sessions are lost when the backend restarts.** There is no persistence
  and no accounts (M8B scope).
