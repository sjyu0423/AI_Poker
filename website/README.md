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

## Architecture

```
website/
  public/screenshots/     real app captures
  src/app/                App Router pages, global CSS
  src/components/         homepage sections
  src/content/site.ts     metrics, links, limitations
  src/lib/paths.ts        GitHub Pages path helpers
  tests/                  content + navigation helper tests
```

Out of scope for W1: deployment workflow, extra pages, interactive play,
and any changes to the research freeze or `web/` app.
