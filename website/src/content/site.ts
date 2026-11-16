/**
 * Central source of truth for the PokerMind portfolio website.
 *
 * Quantitative claims must stay traceable to repository experiment reports.
 * Do not invent metrics, human-player results, or Hold'em claims here.
 */

export const siteMeta = {
  name: "PokerMind",
  tagline: "An AI That Learns How You Play.",
  description:
    "PokerMind is a local-first Leduc Poker AI that models opponent behavior and adapts its strategy using Bayesian inference and game theory.",
  version: "1.0.0",
  researchConfig: "pokermind-research-v1",
  researchDigestPrefix: "a67a2826",
  url: "https://sjyu0423.github.io/AI_Poker/",
  locale: "en_US",
} as const;

export const links = {
  github: "https://github.com/sjyu0423/AI_Poker",
  release: "https://github.com/sjyu0423/AI_Poker/releases/tag/v1.0.0",
  pages: "https://sjyu0423.github.io/AI_Poker/",
  overview: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/overview.md",
  demo: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/demo.md",
  releaseNotes:
    "https://github.com/sjyu0423/AI_Poker/blob/main/docs/release/v1.md",
  researchFreeze:
    "https://github.com/sjyu0423/AI_Poker/blob/main/docs/research/research_freeze.md",
  quickStart:
    "https://github.com/sjyu0423/AI_Poker/blob/main/README.md#quick-start-macos",
  cfrPage: "/research/cfr/",
  cfr: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/concepts/cfr.md",
  m4: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/experiments/m4_leduc_neural.md",
  m5: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/experiments/m5_opponent_modeling.md",
  m6: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/experiments/m6_adaptive_strategy.md",
  m7a: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/experiments/m7a_robust_response.md",
  m7b: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/experiments/m7b_adaptive_reliability.md",
  leduc: "https://github.com/sjyu0423/AI_Poker/blob/main/docs/games/leduc.md",
  appBackend:
    "https://github.com/sjyu0423/AI_Poker/blob/main/docs/design/app_backend.md",
} as const;

export const hero = {
  brand: "PokerMind",
  headline: "An AI That Learns How You Play.",
  summary:
    "PokerMind is a local-first Leduc Poker AI that models opponent behavior and adapts its strategy using Bayesian inference and game theory.",
  primaryCta: { label: "Explore the Research", href: "#research" },
  secondaryCta: { label: "View on GitHub", href: links.github },
  releaseCta: { label: "v1.0.0 release", href: links.release },
  cfrCta: { label: "Interactive CFR explainer", href: links.cfrPage },
  localNote:
    "This site is a research showcase. The playable app runs locally from the GitHub release—not in the browser here.",
} as const;

export const differentiators = [
  {
    id: "observe",
    title: "Learns opponent tendencies",
    plain:
      "As hands unfold, PokerMind watches legal betting choices—fold, call, raise—and builds a statistical picture of how an opponent tends to act in each situation.",
    technical:
      "Bayesian opponent modeling from public actions, with a population prior and uncertainty intervals.",
  },
  {
    id: "adapt",
    title: "Adapts with Restricted Nash Response",
    plain:
      "Instead of chasing every guessed weakness, it solves a carefully restricted game that balances exploitation against staying hard to beat.",
    technical:
      "Restricted Nash Response (RNR) sweeps a trust dial p between the opponent model and a worst-case reply.",
  },
  {
    id: "certify",
    title: "Checks an exact exploitability budget",
    plain:
      "Before a new strategy is used, PokerMind computes exactly how much a perfect counter-strategy could win per hand. Only strategies inside a fixed safety limit go live.",
    technical:
      "Exact exploitability certification: baseline 0.0030 plus budget δ = 0.1 → limit 0.1030 chips/hand.",
  },
] as const;

export const pipelineSteps = [
  {
    step: 1,
    title: "Play hands",
    detail: "Heads-up Leduc Hold'em sessions against the AI.",
  },
  {
    step: 2,
    title: "Observe betting behavior",
    detail: "Only public actions and showdowns are recorded—never folded cards.",
  },
  {
    step: 3,
    title: "Estimate opponent tendencies",
    detail: "A Bayesian model updates fold/call/raise rates with uncertainty.",
  },
  {
    step: 4,
    title: "Compute an adaptive strategy",
    detail: "Restricted Nash Response proposes safer exploiting strategies.",
  },
  {
    step: 5,
    title: "Verify exploitability before use",
    detail: "Exact tree evaluation must pass the budget before installation.",
  },
] as const;

/**
 * Evidence kinds keep portfolio language honest.
 * - exact: computer-checked mathematical facts
 * - measured: simulated-opponent experiments with uncertainty
 * - engineering: local app measurements (machine-dependent timing)
 */
export type EvidenceKind = "exact" | "measured" | "engineering";

export interface ResearchHighlight {
  id: string;
  title: string;
  kind: EvidenceKind;
  value: string;
  detail: string;
  metricDefinition: string;
  sourceLabel: string;
  sourceHref: string;
}

export const researchDisclaimer =
  "All measured adaptive gains and calibration results used simulated Leduc opponents. They are not claims about real human players or Texas Hold'em.";

export const researchHighlights: ResearchHighlight[] = [
  {
    id: "cfr",
    title: "CFR strategy quality",
    kind: "exact",
    value: "0.0030 chips/hand",
    detail:
      "Leduc CFR teacher after 100,000 iterations. Kuhn warm-up reached 0.0010 and matched the known game value (−1/18) to 4×10⁻⁷.",
    metricDefinition:
      "Exploitability = NashConv / 2: how much a perfect counter-strategy wins per hand against this policy.",
    sourceLabel: "docs/concepts/cfr.md · M4 teacher",
    sourceHref: links.cfr,
  },
  {
    id: "neural",
    title: "Neural-network generalization",
    kind: "measured",
    value: "No better than uniform",
    detail:
      "Networks can memorize the CFR table (within ~0.0002 of teacher exploitability on all states), but on unseen states hybrid exploitability was 1.32–1.55 vs 1.41 for uniform random. Neural policies were not adopted in the deployed system.",
    metricDefinition:
      "Hybrid exploitability isolates the cost of predictions on held-out information sets.",
    sourceLabel: "docs/experiments/m4_leduc_neural.md",
    sourceHref: links.m4,
  },
  {
    id: "bayesian",
    title: "Bayesian opponent-model learning",
    kind: "measured",
    value: "~½ at 20 hands, ~¾ at 100",
    detail:
      "Share of the learnable public-action gap closed versus the population prior: about 51% after 20 hands and 76% after 100. Expected calibration error ≤ 0.016, but nominal 90% intervals covered truth only 74% of the time after 20 hands.",
    metricDefinition:
      "Excess log loss vs the best public-information predictor, in nats per opponent decision.",
    sourceLabel: "docs/experiments/m5_opponent_modeling.md",
    sourceHref: links.m5,
  },
  {
    id: "rnr",
    title: "RNR versus mixing baseline",
    kind: "measured",
    value: "+0.131 vs +0.016",
    detail:
      "After 100 hands on held-out simulated opponents (budget δ = 0.1): frozen RNR rule +0.131 chips/hand over CFR; mixing baseline +0.016. Paired frozen − mix = +0.115 [+0.075, +0.155]. Out-of-population styles: frozen +0.619; frozen − mix +0.537 [+0.307, +0.766].",
    metricDefinition:
      "Gain = exact chips/hand above the fixed CFR hero against the true opponent (seats alternate). 95% CIs across opponents.",
    sourceLabel: "docs/experiments/m7a · m7b",
    sourceHref: links.m7b,
  },
  {
    id: "m7b",
    title: "M7B reliability improvements",
    kind: "measured",
    value: "2.3% vs 6.7% losing runs",
    detail:
      "Calibrated 95% lower-confidence selection cut held-out harm rate from 6.7% (point rule) to 2.3%, at a cost of up to ~0.024 chips/hand early that shrinks with more hands. Every played strategy stayed within the exact 0.1030 budget.",
    metricDefinition:
      "Harm rate = share of opponent × seed × hands runs with gain < 0 at δ = 0.1.",
    sourceLabel: "docs/experiments/m7b_adaptive_reliability.md",
    sourceHref: links.m7b,
  },
];

export const guarantees = {
  baselineExploitability: "0.0030 chips/hand",
  budgetLimit: "0.1030 chips/hand",
  budgetDelta: "0.1",
  evaluatorsAgree: "4×10⁻¹⁶",
} as const;

export const screenshots = [
  {
    src: "screenshots/01-new-session.png",
    alt: "PokerMind new-session screen with first-time guidance and AI status panel showing certified exploitability 0.0030.",
    caption: "New session with AI status and safety budget",
  },
  {
    src: "screenshots/02-first-hand.png",
    alt: "First Leduc hand: player private card visible, AI card face down, empty public-card slot, legal actions only.",
    caption: "Playable table with hidden AI card",
  },
  {
    src: "screenshots/03-showdown-result.png",
    alt: "Completed hand showdown with result banner and recent-hands history table.",
    caption: "Showdown result and hand history",
  },
  {
    src: "screenshots/04-learning.png",
    alt: "Table view while a background strategy update shows Learning status; baseline strategy remains in play.",
    caption: "Background learning while play continues",
  },
  {
    src: "screenshots/05-dashboard-observed.png",
    alt: "Learning dashboard observed-counts section with estimate disclaimer for human validation limits.",
    caption: "Observed betting counts",
  },
  {
    src: "screenshots/06-dashboard-tendencies.png",
    alt: "Tendency estimates with 90 percent intervals, typical-player prior ticks, and sparse-data badges.",
    caption: "Estimated tendencies with uncertainty",
  },
  {
    src: "screenshots/07-dashboard-strategy.png",
    alt: "Strategy evolution panel showing certified versions v1 baseline through v3 with exploitability under 0.1030.",
    caption: "Certified strategy version history",
  },
  {
    src: "screenshots/08-mobile-table.png",
    alt: "Mobile layout of the PokerMind Leduc table interface.",
    caption: "Mobile table layout",
  },
] as const;

export const technology = [
  {
    name: "Python & NumPy",
    role: "Core algorithms, exact evaluation, and reproducible experiments.",
  },
  {
    name: "PyTorch",
    role: "Research-only neural distillation studies (M2/M4). Not used as the deployed poker policy.",
  },
  {
    name: "CFR & RNR",
    role: "Game-theoretic baseline and restricted adaptive responses with exact budgets.",
  },
  {
    name: "Bayesian opponent modeling",
    role: "Public-action Dirichlet models with population priors and change detection.",
  },
  {
    name: "FastAPI",
    role: "Local backend: rules, hidden cards, sessions, and certified strategy updates.",
  },
  {
    name: "Next.js & TypeScript",
    role: "Local playable UI and learning dashboard (separate from this static showcase).",
  },
] as const;

export const limitations = [
  "No evidence against real human players; calibration used simulated development opponents only.",
  "Leduc Hold'em only—exact evaluation does not scale to Texas Hold'em.",
  "Uncertainty intervals are not claimed to be calibrated for humans.",
  "Neural networks are research experiments, not the production strategy.",
  "The playable app is local-first; this GitHub Pages site is a static showcase.",
  "RNR solver equilibria are approximate (gap ≤ 0.0031 at the frozen setting); the budget check is exact.",
] as const;

export const finalCta = {
  headline: "Explore the code. Reproduce the experiments. Challenge PokerMind locally.",
  links: [
    { label: "GitHub repository", href: links.github },
    { label: "Published release", href: links.release },
    { label: "Technical documentation", href: links.overview },
  ],
} as const;

export const navItems = [
  { label: "Difference", href: "#difference" },
  { label: "How it works", href: "#how-it-works" },
  { label: "Research", href: "#research" },
  { label: "CFR", href: links.cfrPage },
  { label: "Preview", href: "#preview" },
  { label: "Technology", href: "#technology" },
] as const;
