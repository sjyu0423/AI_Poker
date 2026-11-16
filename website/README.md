# PokerMind website (W1)

Public static portfolio site for PokerMind. This is a research showcase for
GitHub Pages—not the playable local application in `../web/`.

Expected URL: https://sjyu0423.github.io/AI_Poker/

## Stack

- Next.js App Router + TypeScript
- Static export (`output: 'export'`)
- Modern CSS design system (no Tailwind)
- Vitest for content and path helpers

## Base path

GitHub Pages hosts this repository under `/AI_Poker/`. `next.config.ts` sets:

- `basePath` / `assetPrefix` → `/AI_Poker` by default
- `trailingSlash: true`
- `images.unoptimized: true`

Override for a root-domain preview:

```bash
NEXT_PUBLIC_BASE_PATH= npm run build
```

## Commands

```bash
npm ci
npm run dev          # http://localhost:3100/AI_Poker/
npm run typecheck
npm run lint
npm test
npm run build        # writes static files to out/
```

## Content source of truth

All product copy, metrics, limitations, and links live in
`src/content/site.ts`. Quantitative claims are traced to:

- `docs/overview.md`
- `docs/release/v1.md`
- `docs/research/research_freeze.md`
- `docs/experiments/m4_leduc_neural.md`
- `docs/experiments/m5_opponent_modeling.md`
- `docs/experiments/m7a_robust_response.md`
- `docs/experiments/m7b_adaptive_reliability.md`

Screenshots are copies of `docs/screenshots/` in `public/screenshots/`.

## Pages

| route | purpose |
|---|---|
| `/` | W1 homepage showcase |
| `/research/cfr/` | W2 interactive CFR explainer |

Deep links must work under `/AI_Poker/` after static export (trailing slash).

## Architecture

```
website/
  public/screenshots/     real app captures
  src/app/                App Router pages, global CSS
  src/app/research/cfr/   CFR explainer route
  src/components/         homepage + CFR exhibit sections
  src/content/            metrics, links, M1 checkpoint data
  src/lib/                path helpers + regret matching
  tests/                  content, math, and navigation tests
```

Out of scope until later milestones: GitHub Pages deployment workflow,
opponent-modeling exhibit (W3), and any changes to the research freeze or
`web/` app.
