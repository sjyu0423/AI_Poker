# PokerMind website

Public static research showcase for PokerMind, published under GitHub Pages.

**Production URL:** https://sjyu0423.github.io/AI_Poker/

This is **not** the playable local application in `../web/`. The showcase
explains the research stack and ships three interactive exhibits; the Leduc
table, learning dashboard, and strategy installer run only from the local
release.

## Pages

| Route | Purpose |
|---|---|
| `/` | Homepage: product story, research highlights, app preview |
| `/research/cfr/` | Interactive Counterfactual Regret Minimization exhibit |
| `/research/opponent-modeling/` | Interactive Bayesian opponent-modeling exhibit |
| `/research/robust-response/` | Interactive Restricted Nash Response exhibit |

Deep links must keep the trailing slash and the `/AI_Poker/` base path after
static export.

## Stack

- Next.js App Router + TypeScript
- Static export (`output: 'export'`)
- Custom CSS design system (no Tailwind)
- Vitest for content, math, navigation, and SEO checks

## Base path

GitHub Pages hosts this repository under `/AI_Poker/`. `next.config.ts` sets:

- `basePath` / `assetPrefix` → `/AI_Poker` by default
- `trailingSlash: true`
- `images.unoptimized: true`

Override for a root-domain preview:

```bash
NEXT_PUBLIC_BASE_PATH= npm run build
```

## Local development

```bash
cd website
npm ci
npm run dev          # http://localhost:3100/AI_Poker/
npm run typecheck
npm run lint
npm test
```

## Static production build

```bash
cd website
npm ci
npm run build        # writes website/out/
```

Preview the Pages layout locally (symlink required so `/AI_Poker/` resolves):

```bash
mkdir -p /tmp/pokermind-pages
rm -rf /tmp/pokermind-pages/AI_Poker
ln -s "$(pwd)/out" /tmp/pokermind-pages/AI_Poker
cd /tmp/pokermind-pages
python3 -m http.server 3100
# open http://127.0.0.1:3100/AI_Poker/
```

## Content source of truth

Portfolio copy and curated metrics live in `src/content/`. Quantitative claims
must stay traceable to:

- `docs/overview.md`
- `docs/release/v1.md`
- `docs/research/research_freeze.md`
- `docs/experiments/m1…m7b_*.md`
- corresponding `runs/*/metrics.json` / evaluation artifacts

Illustrative educational demos (regret matching, Beta–Binomial learner, RNR
`p` dial) are labeled separately from recorded experiment results.

## Project structure

```
website/
  public/                 screenshots, og-share.jpg, robots.txt, sitemap.xml
  src/app/                App Router pages + global CSS
  src/app/research/       CFR / opponent / RNR exhibit routes
  src/components/         homepage + exhibit UI
  src/content/            copy, links, curated experiment datasets
  src/lib/                path helpers, educational math, metadata helpers
  tests/                  content, math, navigation, SEO, link integrity
```

## GitHub Pages deployment

Workflow: `.github/workflows/website-pages.yml`

On pushes to `main` that touch `website/**` (or via **workflow_dispatch**):

1. `npm ci` in `website/`
2. TypeScript, ESLint, Vitest
3. Static export with `NEXT_PUBLIC_BASE_PATH=/AI_Poker`
4. Upload `website/out/` as a Pages artifact
5. Deploy with official `actions/deploy-pages`

### One-time repository setup (publish)

1. Merge the reviewed website branch into `main` (only when approved).
2. GitHub → **Settings → Pages**
3. **Build and deployment → Source:** GitHub Actions
4. Ensure the `github-pages` environment exists (created automatically on first deploy).
5. Run **Actions → Deploy website to GitHub Pages → Run workflow**, or push a
   website change to `main`.
6. Confirm https://sjyu0423.github.io/AI_Poker/ and deep links under
   `/research/*/`.

The workflow does **not** start FastAPI, deploy the playable app, or use secrets.

## Release-readiness checklist

- [ ] All four routes build as static HTML under `/AI_Poker/`
- [ ] Deep-link refresh works for each exhibit
- [ ] Homepage local-app disclaimer remains visible
- [ ] Metrics cite experiment docs; toys labeled illustrative
- [ ] No human-player or Texas Hold’em performance claims
- [ ] `robots.txt`, `sitemap.xml`, and `og-share.jpg` resolve under `/AI_Poker/`
- [ ] 404 page links back to homepage and exhibits
- [ ] Keyboard focus visible; reduced-motion respected
- [ ] Typecheck, ESLint, Vitest, and `npm run build` pass
- [ ] Pages source set to GitHub Actions; workflow green on `main`

## Known limitations

- Static showcase only—no playable poker in the browser
- Measured adaptive gains use simulated Leduc opponents
- Educational interactions are not production solvers
- Exact evaluation does not scale to Texas Hold’em
- Publishing requires an approved merge to `main` plus Pages configuration
